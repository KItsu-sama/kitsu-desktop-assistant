# core//kitsu_core.py

import asyncio
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from core.personality.emotion_engine import EmotionEngine
from core.llm.prompt_builder import PromptBuilder   
from data.templates.base_personal import character_context
from core.meta.meta_controller import MetaController, ResponseMode
from rich.console import Console

console = Console()


log = logging.getLogger(__name__)

class KitsuIntegrated:
    """
    Kitsu with emotion system integrated
    """
    
    def __init__(
        self,
        model: str = "gemma:2b",
        temperature: float = 0.8,
        templates_path: Path = None,
        streaming: bool = True,
        continuous_decay: bool = True
    ):
        self.model = model
        self.temperature = temperature
        self.templates_path = templates_path or Path("data/templates")
        self.streaming = streaming
        self.continuous_decay = continuous_decay
        self._build_greeting_prompt = PromptBuilder._build_greeting_prompt
        self.meta_controller = MetaController()

        # Character context
        self.character_context = character_context

        # Components (initialized later)
        self.memory = None
        self.planner = None  # planner system
        self.llm = None
        self.kitsu_self = None
        self.emotion_engine = None

        # Dev console (initialized later)
        self.dev_router = None  # Renamed for clarity

        # State
        self.running = False
        self._decay_task = None
        # Greet on startup flag (set via config)
        self.greet_on_startup: bool = True


    async def initialize(self):
        """Initialize all components"""
        log.info("🦊 Initializing Kitsu...")
        
        try:
            # Initialize KitsuSelf first (so memory can reference it)
            from core.personality.kitsu_self import KitsuSelf
            
            self.kitsu_self = KitsuSelf()
            log.info("✅ KitsuSelf initialized")
            
        except Exception as e:
            log.error(f"❌ KitsuSelf initialization failed: {e}")
            raise
        
        try:
            # Initialize memory system (with plugins)
            from core.memory.memory_manager import MemoryManager, MemoryConfig
            
            # Create config
            memory_config = MemoryConfig(
                max_history=200,
                auto_save=True,
                save_interval=300,  # 5 minutes
                compression_enabled=True
            )
            
            self.memory = MemoryManager(
                kitsu_self=self.kitsu_self,
                memory_path=Path("data/memory/memory.json"),
                config=memory_config
            )
            log.info("✅ Memory system initialized (with plugins)")
            
        except Exception as e:
            log.error(f"❌ Memory initialization failed: {e}")
            raise
        
        try:
            # Initialize LLM with templates path
            from core.llm.llm_interface import LLMInterface
            
            self.llm = LLMInterface(
                model=self.model,
                temperature=self.temperature,
                character_context=self.character_context,
                memory_manager=self.memory,
                templates_path=self.templates_path,
                streaming=self.streaming
            )
            log.info(f"✅ LLM initialized (templates: {self.templates_path})")
            
        except Exception as e:
            log.error(f"❌ LLM initialization failed: {e}")
            raise
        
        try:
            # Initialize emotion engine
            from core.personality.emotion_engine import EmotionEngine
            
            self.emotion_engine = EmotionEngine(
                kitsu_self=self.kitsu_self,
                triggers_path=Path("data/triggers.json"),
                continuous_decay=self.continuous_decay
            )
            
            # Link emotion engine to KitsuSelf
            self.kitsu_self.set_emotion_engine(self.emotion_engine)
            
            log.info("✅ Emotion engine initialized")
            
        except Exception as e:
            log.error(f"❌ Emotion engine initialization failed: {e}")
            raise
        
        # Start emotion decay loop
        self._decay_task = asyncio.create_task(
            self.emotion_engine.run(),
            name="emotion_decay"
        )
        
        # Initialize Online RL Engine
        try:
            from core.learning.online_rl import OnlineRLEngine
            
            self.rl_engine = OnlineRLEngine(
                save_path=Path("data/learning/online_rl.json")
            )
            log.info("✅ Online RL engine initialized")
        except Exception as e:
            log.error(f"❌ RL engine initialization failed: {e}")
            self.rl_engine = None


        log.info("🎉 Kitsu is ready!\n")

        # --- Initialize dev console router (optional) ---
        try:
            # import here to avoid hard dependency if dev_console is missing
            from core.dev.console_router import ConsoleRouter

            # minimal module registry for /reset_module and debug usage
            module_registry = {
                "memory": self.memory,
                "llm": self.llm,
                "emotion_engine": self.emotion_engine,
            }

            # attach fallback manager if available (optional)
            try:
                from core.fallback_manager import FallbackManager
                # assume you will instantiate fallback manager elsewhere;
                # if not present, create a lightweight one
                if not getattr(self, "fallback", None):
                    self.fallback = FallbackManager(memory=self.memory)
                module_registry["fallback"] = self.fallback
            except Exception:
                # fallback manager optional; ignore if not present
                pass

            self.console_router = ConsoleRouter(
                memory=self.memory,
                logger=log,
                modules=module_registry
            )
            log.info("✅ Dev console router initialized")
        except Exception as e:
            log.warning(f"Dev console not available: {e}")


        try:
            from core.dev.console_router import ConsoleRouter
            
            # Build module registry for hot-reload support
            module_registry = {
                "memory": self.memory,
                "llm": self.llm,
                "emotion_engine": self.emotion_engine,
                "kitsu_self": self.kitsu_self,
            }
            
            # Add fallback manager if available
            if hasattr(self, 'fallback') and self.fallback:
                module_registry["fallback"] = self.fallback
            
            # Custom admin check using memory system
            def admin_check(user_id: str) -> bool:
                """Check if user is admin using memory system."""
                try:
                    user_info = self.memory.get_user_info()
                    # Check permissions in user profile
                    perms = user_info.get("permissions", {})
                    return perms.get("admin", False) or perms.get("dev_console", False)
                except Exception:
                    # Fallback to hardcoded list
                    return user_id in ["Zino", "Natadaide"]
            
            self.dev_router = ConsoleRouter(
                memory=self.memory,
                logger=log,
                modules=module_registry,
                admin_check=admin_check
            )
            
            log.info("✅ Dev console router initialized")
            
        except ImportError as e:
            log.warning(f"⚠️  Dev console not available (module not found): {e}")
            self.dev_router = None
        except Exception as e:
            log.error(f"❌ Dev console initialization failed: {e}")
            self.dev_router = None

    
    async def process_input(self, user_input: str) -> str:
        """
        Process user input with emotion system
        """
        try:
            # Step 1: Analyze emotion (using LLM)
            emotion_analysis = await self.llm.analyze_emotion(user_input)
            log.debug(f"Emotion analysis: {emotion_analysis}")
            
            # Step 2: Check for triggers
            trigger = emotion_analysis.get("trigger")
            if trigger and self.emotion_engine.trigger_manager:
                if self.emotion_engine.trigger_manager.can_fire(trigger):
                    log.info(f"🔥 Firing trigger: {trigger}")
                    self.emotion_engine.fire_trigger(trigger)
            
            # Step 3: Add emotion to stack (from HF emotion model)
            hf_emotion = emotion_analysis.get("hf_emotion", "neutral")
            sentiment = emotion_analysis.get("sentiment", "neutral")
            
            # Map sentiment to intensity
            intensity_map = {
                "positive": 0.7,
                "neutral": 0.3,
                "negative": 0.6
            }
            intensity = intensity_map.get(sentiment, 0.5)
            
            self.emotion_engine.add_emotion(
                name=hf_emotion,
                intensity=intensity,
                duration=5.0
            )
            
            # Step 4: Tick emotion engine (update personality)
            await self.emotion_engine.tick()
            
            # Step 5: Get current state
            state = self.emotion_engine.get_state_dict()
            mood = state["mood"]
            style = state["style"]
            
            # Log state if changed
            dominant = state["dominant_emotion"]
            log.info(f"😊 {mood}/{style} ({dominant})")


            # Step 6: Meta decision
            user_info = self.memory.get_user_info() if self.memory else {}
            permissions = user_info.get("permissions", {})

            mode = self.meta_controller.decide(
                emotion_state=state,
                emotion_analysis=emotion_analysis,
                user_permissions=permissions
            )

            if mode == ResponseMode.SILENT:
                return "..."

            
            # Get RL preferences ===================== fine tuning =================
            preferences = self.rl_engine.get_user_preferences() if self.rl_engine else None
        
            
            # Step 7: Generate response
            response = await self.llm.generate_response(
                user_input=user_input,
                mood=mood,
                style=style,
                preferences=preferences,
                mode=mode.value,
                stream=False
            )

            # Record interaction for learning
            if self.rl_engine:
                context = {
                    "mood": mood,
                    "style": style,
                    "intent": emotion_analysis.get("intent"),
                    "user_input": user_input
                }
                # Reward will be calculated on next user response
            self.rl_engine.record_interaction(context, response)
            
            # Step 8: Save to memory (using your plugin system!)
            self.memory.remember("user", user_input, emotion=dominant)
            self.memory.remember("kitsu", response, emotion=dominant)
            
            return response
            
        except Exception as e:
            log.exception(f"Error processing input: {e}")
            return "Sorry, I'm having trouble thinking right now... 🦊"
        
    async def greet_user(self):
        """Generate initial greeting when Kitsu starts up."""
        try:
            # Get user info
            user_info = self.memory.get_user_info() if self.memory else {}
            name = user_info.get("nickname") or user_info.get("name", "there")
            title = user_info.get("refer_title") or name
            
            # Get emotional state
            state = self.emotion_engine.get_state_dict()
            mood = state["mood"]
            style = state["style"]
            
            # --- Build prompt ---
            if getattr(self.llm, "prompt_builder", None):
                greeting_prompt = (
                    self.llm.prompt_builder
                    ._build_greeting_prompt(title, mood, style)
                )
            else:
                # fallback: create PromptBuilder manually
                from core.llm.prompt_builder import PromptBuilder
                pb = PromptBuilder(
                    character_context=self.character_context,
                    memory_manager=self.memory
                )
                greeting_prompt = pb._build_greeting_prompt(title, mood, style)

            # --- Generate greeting ---
            greeting = await self.llm.generate_response(
                user_input="",
                mood=mood,
                style=style,
                stream=False,
                custom_prompt=greeting_prompt
            )

            console.print(f"[magenta]Kitsu:[/magenta] {greeting}\n")

            # Save memory
            self.memory.remember("kitsu", greeting, emotion="happy")

        except Exception as e:
            log.error(f"Failed to greet: {e}")
            fallback = "Hiya~! I'm awake~! 🦊✨"
            console.print(f"[magenta]Kitsu:[/magenta] {fallback}\n")

    
    async def chat_loop(self):
        """Main chat loop"""
        self.running = True
        
        # Keep chat banner minimal — main.py prints the prominent startup Panel
        console.rule("[bold green]🦊 KITSU - Desktop VTuber Assistant[/bold green]")

        # Generate greeting before chat loop starts (honor suppression flag)
        if getattr(self, 'greet_on_startup', True):
            await self.greet_user()
        
        while self.running:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    input,
                    "\n\033[94mYou:\033[0m "
                )
                
                user_input = user_input.strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue
                
                # Process input
                response = await self.process_input(user_input)
                console.print(f"[bold magenta]Kitsu:[/bold magenta] {response}" , soft_wrap=True)

                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                log.exception(f"Error in chat loop: {e}")
        
        console.print("\n\n👋 Kitsu: Bye bye! See you later! 🦊\n")
    


    async def _handle_command(self, command: str):
        """Handle special commands"""
        parts = command.strip().split()
        cmd = parts[0].lower()
        
        # SYSTEM COMMANDS
        # ============================================================
        
        if cmd in ["/quit", "/exit"]:
            self.running = False
            return
        
        elif cmd == "/reload":
            self.llm.reload_templates()
            console.print("🔄 Mode templates reloaded!")
            return
        
        elif cmd == "/clear":
            self.memory.clear()
            console.print("💭 Memory cleared")
            return
        
        elif cmd == "/optimize":
            self.memory.optimize_memory()
            stats = self.memory.get_stats()
            console.print(f"🧹 Memory optimized: {stats['total_sessions']} sessions")
            return
        
        # INFO COMMANDS
        # ============================================================
        
        elif cmd == "/stats":
            stats = self.memory.get_stats()
            console.print(f"\n📊 Memory Statistics:")
            console.print(f"  Total sessions: {stats['total_sessions']}")
            console.print(f"  Emotions: {stats['emotion_distribution']}")
            console.print(f"  Memory usage: {stats['memory_usage_bytes'] / 1024:.1f} KB")
            console.print(f"  Plugins: {', '.join(stats['active_plugins'])}")
            return
        
        elif cmd == "/state":
            state = self.emotion_engine.get_state_dict()
            console.print(f"\n📊 Current State:")
            console.print(f"  Mood: {state['mood']}")
            console.print(f"  Style: {state['style']}")
            console.print(f"  Mode (legacy): {state['current_mode']}")
            console.print(f"  Dominant emotion: {state['dominant_emotion']}")
            console.print(f"  Hidden: {state['is_hidden']}")
            console.print(f"  Stack size: {state['stack_size']}")
            console.print(f"  Avatar: {self.emotion_engine.get_avatar_hint()}")
            return
        
        elif cmd == "/llm":
            is_available = self.llm._check_availability() if hasattr(self.llm, '_check_availability') else True
            console.print(f"🤖 LLM Status: {'✅ Available' if is_available else '❌ Unavailable'}")
            return
        
        # SEARCH COMMAND
        # ============================================================
        
        elif cmd == "/search":
            if len(parts) < 2:
                print("❌ Usage: /search <query>")
                return
            
            query = " ".join(parts[1:])
            results = self.memory.search(query, limit=5)
            print(f"\n🔍 Search results for '{query}':")
            
            if not results:
                print("  No results found.")
                return
            
            for i, result in enumerate(results, 1):
                role = result.get('role', 'unknown')
                text = result.get('text', '')
                emotion = result.get('emotion', 'neutral')
                score = result.get('relevance_score', 0)
                print(f"  {i}. [{role}] ({emotion}, score: {score:.2f})")
                print(f"     {text[:80]}...")
            return
        

        # PERSONALITY COMMANDS
        # ============================================================
            """
            elif cmd == "/mood":
                if len(parts) < 2:
                    print("❌ Usage: /mood <behave|mean|flirty>")
                    return
                
                mood = parts[1].lower()
                if mood in ["behave", "mean", "flirty"]:
                    self.emotion_engine.set_mood(mood)
                    print(f"✨ Mood set to: {mood}")
                else:
                    print("❌ Invalid mood. Use: behave, mean, or flirty")
                return"""
        
        # debugging mood command
        elif cmd == "/mood":
            if len(parts) < 2:
                print("❌ Usage: /mood <behave|mean|flirty>")
                return
            
            mood = parts[1].lower()
            # Support special 'clear' action
            if mood == 'clear':
                if self.emotion_engine:
                    self.emotion_engine.clear_mood_override()
                    print("✨ Manual mood override cleared")
                else:
                    print("⚠️  No emotion engine available")
                return

            # Check for persist token and duration
            persist = ('--persist' in parts) or ('persist' in parts)
            duration = None
            for p in parts[2:]:
                if p.startswith('duration='):
                    try:
                        duration = float(p.split('=',1)[1])
                    except Exception:
                        duration = None
            if mood in ["behave", "mean", "flirty"]:
                # Show before state
                before = self.emotion_engine.get_state_dict()
                print(f"🔍 Before: mood={before['mood']}, style={before['style']}")
                
                # Set mood
                # Use duration if provided
                if duration is not None:
                    self.emotion_engine.set_mood(mood, duration=duration, persist=persist)
                else:
                    self.emotion_engine.set_mood(mood, persist=persist)
                
                # Show after state (immediate)
                after = self.emotion_engine.get_state_dict()
                print(f"🔍 After:  mood={after['mood']}, style={after['style']}")
                
                # Verify the internal state
                print(f"🔍 Direct: emotion_engine.mood = {self.emotion_engine.mood}")
                
                if after['mood'] == mood:
                    print(f"✨ Mood set to: {mood}")
                else:
                    print(f"⚠️  ISSUE DETECTED!")
                    print(f"   Expected mood: {mood}")
                    print(f"   Actual mood: {after['mood']}")
                    print(f"   Direct access: {self.emotion_engine.mood}")
            else:
                print("❌ Invalid mood. Use: behave, mean, or flirty")
            return
        
        elif cmd == "/style":
            if len(parts) < 2:
                print("❌ Usage: /style <chaotic|sweet|cold|silent>")
                return
            
            style = parts[1].lower()
            if style in ["chaotic", "sweet", "cold", "silent"]:
                self.emotion_engine.set_style(style)
                print(f"✨ Style set to: {style}")
            else:
                print("❌ Invalid style. Use: chaotic, sweet, cold, or silent")
            return
        
        elif cmd == "/trigger":
            if len(parts) < 2:
                print("❌ Usage: /trigger <trigger_name>")
                return
            
            trigger_name = parts[1]
            print(f"🔥 Firing trigger: {trigger_name}")
            self.emotion_engine.fire_trigger(trigger_name)
            await self.emotion_engine.tick()
            state = self.emotion_engine.get_state_dict()
            print(f"State: {state['mood']}/{state['style']} ({state['dominant_emotion']})")
            return
        
        elif cmd == "/hide":
            self.emotion_engine.hide()
            print("😴 Kitsu is now hidden")
            return
        
        elif cmd == "/unhide":
            self.emotion_engine.unhide()
            print("👋 Kitsu is awake!")
            return
        

        # USER MANAGEMENT
        # ============================================================
        
        elif cmd == "/user":
            await self._handle_user_command(parts, command)
            return
        

        # DEVELOPER CONSOLE COMMANDS
        # ============================================================
        
        elif cmd in ["/train", "/rate", "/errors", "/debug", "/simulate_error", 
                    "/export_logs", "/reset_module", "/dev_stats"]:
            
            if not self.dev_router:
                print("❌ Dev console not initialized or not available.")
                return
            
            # Get current user identity
            try:
                user_info = self.memory.get_user_info()
                current_user = user_info.get("name") or user_info.get("nickname") or "unknown"
            except Exception:
                current_user = "unknown"
            
            # Route the command (strip leading slash)
            try:
                result = self.dev_router.route(
                    command=command.lstrip('/'),
                    user=current_user
                )
                
                ok = result.get("ok", False)
                res = result.get("result", "")
                
                # Pretty print result
                if isinstance(res, str):
                    print(f"{'✅' if ok else '❌'} {res}")
                else:
                    import json
                    print(f"{'✅' if ok else '❌'} {json.dumps(res, indent=2)}")
                    
            except Exception as e:
                log.exception(f"Dev command error: {e}")
                print(f"❌ Failed to execute dev command: {e}")
            
            return
        

        # HELP
        # ============================================================
        
        elif cmd in ["/help", "/h"]:
            self._print_help()
            return
        
        # (moved) runtime setup handler inserted above
        elif cmd in ["/first_meet", "/firstmeet", "/first-meet"]:
            # Run interactive setup wizard if terminal is interactive, otherwise apply defaults
            try:
                import sys
                from scripts.setup_wizard import SetupWizard
                wiz = SetupWizard()

                if sys.stdin and sys.stdin.isatty():
                    # run interactive wizard in an executor so we don't block the event loop
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, wiz.run)
                else:
                    # Non-interactive environment, write defaults directly
                    wiz.apply_defaults()

                # Reload minimal configuration and apply to the current instance
                try:
                    import json
                    cfg_path = Path("data/config.json")
                    if cfg_path.exists():
                        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
                        self.greet_on_startup = cfg.get('greet_on_startup', self.greet_on_startup)
                except Exception as re:
                    log.warning(f"Reloading config after setup wizard failed: {re}")

                # Apply personality defaults if present
                try:
                    persona_path = Path("data/config/personality.json")
                    if persona_path.exists():
                        import json
                        persona = json.loads(persona_path.read_text(encoding='utf-8'))
                        default_mood = persona.get('default_mood')
                        default_style = persona.get('default_style')
                        if default_mood and self.emotion_engine:
                            # persist default as a manual override so the running session honors it
                            self.emotion_engine.set_mood(default_mood, persist=True)
                        if default_style and self.emotion_engine:
                            self.emotion_engine.set_style(default_style)
                except Exception as pe:
                    log.warning(f"Applying personality defaults failed: {pe}")

                console.print("[green]✅ Setup wizard applied (runtime).[/green]")
            except Exception as e:
                log.exception(f"Failed to run setup wizard at runtime: {e}")
                print(f"❌ /first_meet failed: {e}")
            return
        

        # UNKNOWN COMMAND
        # ============================================================
        
        else:
            print(f"❌ Unknown command: {cmd}")
            print("   [cyan]Type[/cyan] /help for available commands")
            return

        # stray /first_meet block removed (handled earlier)


    async def _handle_user_command(self, parts: List[str], full_command: str):
        """Handle /user subcommands (extracted for clarity)."""
        
        # Show info
        if len(parts) == 1:
            info = self.memory.get_user_info()
            print("\n📊 \033[94mUser Info:\033[0m")
            print(f"  Name: {info.get('name', 'Unknown')}")
            print(f"  Gender: {info.get('gender', 'Unknown')}")
            print(f"  Nickname: {info.get('nickname', 'Unknown')}")
            print(f"  Title (Kitsu calls you): {info.get('refer_title', 'Unknown')}")
            print(f"  Status: {info.get('status', 'Unknown')}")
            
            rel = info.get("relationship", {})
            print(f"  Relationship: trust={rel.get('trust_level')}, affinity={rel.get('affinity')}, lore='{rel.get('lore_tag', '')}'")
            
            perms = info.get("permissions", {})
            print("  Permissions:")
            for k, v in perms.items():
                print(f"    {k}: {v}")
            print("")
            return
        
        sub = parts[1].lower()
        
        # /user set
        if sub == "set":
            if len(parts) < 4:
                print("❌ Usage: /user set <field> <value>")
                return
            
            raw_field = parts[2].lower()
            
            # normalize mapping
            FIELD_MAP = {
                "name": "name",
                "nickname": "nickname",
                "nick": "nickname",
                "title": "refer_title",
                "refer_title": "refer_title",
                "gender": "gender",
                "status": "status",
                "role": "status",
            }
            
            field = FIELD_MAP.get(raw_field, raw_field)
            
            # value with spacing preserved - use full_command instead of command
            raw_parts = full_command.strip().split(" ", 3)
            if len(raw_parts) < 4:
                print("❌ Missing value. Usage: /user set <field> <value>")
                return
            
            value = raw_parts[3]
            
            # strip quotes
            if (value.startswith('"') and value.endswith('"')) or \
                (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            
            # parse numeric or boolean
            if value.lower() in ["true", "false"]:
                parsed_value = (value.lower() == "true")
            else:
                try:
                    parsed_value = float(value) if "." in value else int(value)
                except:
                    parsed_value = value
            
            # ------------------ VALIDATION ------------------
            # only allow known fields
            valid_root_fields = {
                "name", "nickname", "refer_title", "gender", "status",
                "permissions"
            }
            
            valid_relationship_fields = {
                "relationship.trust_level",
                "relationship.affinity",
                "relationship.lore_tag",
            }
            
            if not (
                field in valid_root_fields or
                field in valid_relationship_fields or
                field.startswith("permissions.")
            ):
                print(f"❌ Unknown or locked field: {field}")
                return
            
            # ------------------ BUILD UPDATE ------------------
            if field.startswith("permissions."):
                _, key = field.split(".", 1)
                update_map = {"permissions": {key: parsed_value}}
            else:
                update_map = {field: parsed_value}
            
            try:
                self.memory.set_user_info(**update_map)
                self.memory.save_user()
                print(f"✅ Updated {field} -> {parsed_value}")
            except Exception as e:
                print(f"❌ Failed to update user info: {e}")
            return
        
        # /user reset
        elif sub == "reset":
            target = parts[2].lower() if len(parts) >= 3 else "all"
            
            if target not in ("profile", "permissions", "all"):
                print("❌ Invalid reset target. Use: profile | permissions | all")
                return
            
            self.memory.reset_user_info(None if target == "all" else target)
            self.memory.save_user()
            print(f"🔁 Reset {target} to defaults")
            return
        
        else:
            print(f"❌ Unknown /user subcommand: {sub}")
            print("   Usage: /user | /user set <field> <value> | /user reset [profile|permissions|all]")
            return


    def _print_help(self):
        """Print help message."""
        print("\n" + "="*60)
        print("  🦊 KITSU COMMANDS")
        print("="*60)
        print("\n📁 System:")
        print("  /quit, /exit       - Exit Kitsu")
        print("  /reload            - Reload templates")
        print("  /clear             - Clear memory")
        print("  /optimize          - Optimize memory")
        print("\n📊 Information:")
        print("  /stats             - Show memory statistics")
        print("  /state             - Show emotional state")
        print("  /llm               - Show LLM status")
        print("  /search <query>    - Search memory")
        print("\n😊 Personality:")
        print("  /mood <mode>       - Set mood (behave|mean|flirty)")
        print("  /style <style>     - Set style (chaotic|sweet|cold|silent)")
        print("  /trigger <name>    - Fire emotion trigger")
        print("  /hide              - Hide Kitsu")
        print("  /unhide            - Show Kitsu")
        print("\n👤 User:")
        print("  /user              - Show user info")
        print("  /user set <field> <value>  - Update user field")
        print("  /user reset [target]       - Reset user data")
        print("\n🛠️  Developer (Admin Only):")
        print("  /train <text>      - Save response override for training")
        print("  /rate <1-5>        - Rate last response")
        print("  /errors [n]        - Show last n errors")
        print("  /debug             - Show debug summary")
        print("  /simulate_error    - Test fallback system")
        print("  /export_logs       - Export error logs")
        print("  /reset_module <name> - Hot-reload a module")
        print("  /dev_stats         - Show system statistics")
        print("  /first_meet        - Re-run the setup wizard (interactive or non-interactive)")
        print("\n" + "="*60 + "\n")
        
    
    async def shutdown(self):
        """Cleanup"""
        self.running = False
        
        # Cancel emotion decay task
        if self._decay_task and not self._decay_task.done():
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass
        
        # Save KitsuSelf state
        if self.kitsu_self:
            try:
                memory_path = Path("data/memory/kitsu_state.json")
                memory_path.parent.mkdir(parents=True, exist_ok=True)
                self.kitsu_self.save_state(memory_path)
                log.info("State saved")
            except Exception as e:
                log.warning(f"Failed to save state: {e}")
        
        log.info("Shutting down Kitsu...")

        # --- Dev-console cleanup placeholder ---
        # During shutdown we previously attempted to route developer commands
        # but there was an undefined variable. Keep a safe, explicit no-op
        # here to avoid NameError while preserving the hook for future work.
        if getattr(self, 'console_router', None):
            try:
                # Potential place to flush dev console tasks in future
                pass
            except Exception:
                pass

