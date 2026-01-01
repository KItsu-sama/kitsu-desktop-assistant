# =============================================================================
# FILE: scripts/test_lora_system.py
# Complete test suite for LoRA integration
# =============================================================================

import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class LoRASystemTester:
    """Complete test suite for LoRA system"""
    
    def __init__(self):
        self.results = []
        self.kitsu = None
    
    async def setup(self):
        """Initialize Kitsu for testing"""
        console.print("\n[cyan]Initializing Kitsu for testing...[/cyan]")
        
        try:
            from core.kitsu_core import KitsuIntegrated
            
            self.kitsu = KitsuIntegrated(
                model="tinyllama",
                temperature=0.8,
                streaming=False
            )
            
            await self.kitsu.initialize()
            console.print("[green]✓[/green] Kitsu initialized\n")
            return True
            
        except Exception as e:
            console.print(f"[red]✗[/red] Setup failed: {e}\n")
            return False
    
    def test_result(self, name: str, passed: bool, details: str = ""):
        """Record test result"""
        self.results.append({
            "name": name,
            "passed": passed,
            "details": details
        })
        
        icon = "[green]✓[/green]" if passed else "[red]✗[/red]"
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        
        console.print(f"{icon} {name}: {status}")
        if details:
            console.print(f"   {details}")
    
    # =========================================================================
    # Test 1: Manager Initialization
    # =========================================================================
    
    def test_manager_init(self):
        """Test LoRA manager is properly initialized"""
        console.print("\n[bold]Test 1: Manager Initialization[/bold]")
        
        try:
            llm = self.kitsu.llm
            
            # Check manager exists
            if not hasattr(llm, 'lora_manager'):
                self.test_result(
                    "Manager existence",
                    False,
                    "lora_manager not found on LLMInterface"
                )
                return
            
            manager = llm.lora_manager
            
            if manager is None:
                self.test_result(
                    "Manager existence",
                    False,
                    "lora_manager is None"
                )
                return
            
            self.test_result("Manager existence", True)
            
            # Check adapters discovered
            stats = manager.get_stats()
            adapter_count = stats.get("total_adapters", 0)
            
            self.test_result(
                "Adapter discovery",
                adapter_count > 0,
                f"Found {adapter_count} adapters"
            )
            
            # Check required styles
            available = stats.get("available_styles", [])
            required = ["chaotic", "sweet", "cold", "silent"]
            
            missing = [s for s in required if s not in available]
            
            self.test_result(
                "Required styles",
                len(missing) == 0,
                f"Available: {available}, Missing: {missing}" if missing else f"All present: {available}"
            )
            
        except Exception as e:
            self.test_result("Manager initialization", False, str(e))
    
    # =========================================================================
    # Test 2: Emotion Mapping
    # =========================================================================
    
    def test_emotion_mapping(self):
        """Test emotion state → LoRA style mapping"""
        console.print("\n[bold]Test 2: Emotion Mapping[/bold]")
        
        try:
            manager = self.kitsu.llm.lora_manager
            
            # Test cases: (emotion_state, expected_style)
            test_cases = [
                (
                    {"mood": "behave", "style": "chaotic", "dominant_emotion": "playful"},
                    "chaotic"
                ),
                (
                    {"mood": "behave", "style": "sweet", "dominant_emotion": "happy"},
                    "sweet"
                ),
                (
                    {"mood": "mean", "style": "cold", "dominant_emotion": "hurt"},
                    "cold"
                ),
                (
                    {"mood": "behave", "style": "silent", "dominant_emotion": "tired"},
                    "silent"
                ),
            ]
            
            for emotion_state, expected in test_cases:
                result = manager.select_for_emotion(emotion_state)
                
                # result is None if already on target style
                # so we check current style
                if result is None:
                    actual = manager.current_style
                else:
                    actual = result
                
                passed = actual == expected
                
                self.test_result(
                    f"Map {emotion_state['style']} → {expected}",
                    passed,
                    f"Got: {actual}"
                )
            
        except Exception as e:
            self.test_result("Emotion mapping", False, str(e))
    
    # =========================================================================
    # Test 3: Runtime Switching
    # =========================================================================
    
    def test_runtime_switching(self):
        """Test runtime LoRA adapter switching"""
        console.print("\n[bold]Test 3: Runtime Switching[/bold]")
        
        try:
            manager = self.kitsu.llm.lora_manager
            
            # Record initial state
            initial = manager.current_style
            
            # Test switch to each style
            styles = ["chaotic", "sweet", "cold", "silent"]
            
            for style in styles:
                success = manager.switch_adapter(style, force=True)
                
                self.test_result(
                    f"Switch to {style}",
                    success and manager.current_style == style,
                    f"Current: {manager.current_style}"
                )
                
                time.sleep(0.1)  # Small delay
            
            # Check switch count
            stats = manager.get_stats()
            switches = stats.get("switch_count", 0)
            
            self.test_result(
                "Switch count tracking",
                switches >= len(styles),
                f"Recorded {switches} switches"
            )
            
            # Restore initial
            manager.switch_adapter(initial, force=True)
            
        except Exception as e:
            self.test_result("Runtime switching", False, str(e))
    
    # =========================================================================
    # Test 4: Automatic Integration
    # =========================================================================
    
    async def test_automatic_integration(self):
        """Test automatic LoRA switching during generation"""
        console.print("\n[bold]Test 4: Automatic Integration[/bold]")
        
        try:
            # Set specific emotion state
            emotion_engine = self.kitsu.emotion_engine
            emotion_engine.set_mood("mean")
            emotion_engine.set_style("cold")
            
            # Record initial LoRA
            manager = self.kitsu.llm.lora_manager
            initial = manager.current_style
            
            # Trigger generation (should auto-switch)
            response = await self.kitsu.process_input("Hello")
            
            # Check if LoRA switched
            final = manager.current_style
            
            # For mean/cold, should use "cold" adapter
            expected = "cold"
            
            self.test_result(
                "Auto-switch on generation",
                final == expected or initial != final,
                f"Initial: {initial}, Final: {final}, Expected: {expected}"
            )
            
            # Reset to neutral
            emotion_engine.set_mood("behave")
            emotion_engine.set_style("chaotic")
            
        except Exception as e:
            self.test_result("Automatic integration", False, str(e))
    
    # =========================================================================
    # Test 5: Performance
    # =========================================================================
    
    def test_performance(self):
        """Test performance impact of switching"""
        console.print("\n[bold]Test 5: Performance[/bold]")
        
        try:
            manager = self.kitsu.llm.lora_manager
            
            # Measure switch time
            times = []
            
            for _ in range(10):
                start = time.time()
                manager.switch_adapter("chaotic", force=True)
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = sum(times) / len(times)
            
            # Should be very fast (< 100ms)
            self.test_result(
                "Switch speed",
                avg_time < 0.1,
                f"Average: {avg_time*1000:.2f}ms"
            )
            
            # Check memory usage (basic check)
            stats = manager.get_stats()
            adapters = stats.get("total_adapters", 0)
            
            # Each LoRA is ~16MB, should be manageable
            estimated_memory = adapters * 16  # MB
            
            self.test_result(
                "Memory usage",
                estimated_memory < 200,
                f"Estimated: {estimated_memory}MB for {adapters} adapters"
            )
            
        except Exception as e:
            self.test_result("Performance", False, str(e))
    
    # =========================================================================
    # Test 6: Fallback Behavior
    # =========================================================================
    
    def test_fallback_behavior(self):
        """Test graceful fallback when adapters unavailable"""
        console.print("\n[bold]Test 6: Fallback Behavior[/bold]")
        
        try:
            manager = self.kitsu.llm.lora_manager
            
            # Try to switch to non-existent style
            success = manager.switch_adapter("nonexistent_style")
            
            self.test_result(
                "Reject invalid style",
                not success,
                "Correctly rejected nonexistent style"
            )
            
            # Check system still works
            current = manager.current_style
            
            self.test_result(
                "System stability",
                current is not None,
                f"Still on valid style: {current}"
            )
            
        except Exception as e:
            self.test_result("Fallback behavior", False, str(e))
    
    # =========================================================================
    # Test 7: Config Persistence
    # =========================================================================
    
    def test_config_persistence(self):
        """Test configuration saves and loads correctly"""
        console.print("\n[bold]Test 7: Config Persistence[/bold]")
        
        try:
            manager = self.kitsu.llm.lora_manager
            
            # Switch to specific style
            target = "sweet"
            manager.switch_adapter(target, force=True)
            
            # Check config file
            config_path = Path("data/config.json")
            
            if not config_path.exists():
                self.test_result(
                    "Config file exists",
                    False,
                    "data/config.json not found"
                )
                return
            
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check style saved
            saved_style = config.get("model", {}).get("style")
            
            self.test_result(
                "Config persistence",
                saved_style == target,
                f"Saved: {saved_style}, Expected: {target}"
            )
            
        except Exception as e:
            self.test_result("Config persistence", False, str(e))
    
    # =========================================================================
    # Run All Tests
    # =========================================================================
    
    async def run_all(self):
        """Run complete test suite"""
        console.print(Panel.fit(
            "[bold magenta]🦊 KITSU LORA SYSTEM TEST SUITE[/bold magenta]\n"
            "[dim]Validating emotion-driven adapter switching[/dim]",
            border_style="magenta"
        ))
        
        # Setup
        if not await self.setup():
            console.print("\n[red]Setup failed, cannot continue[/red]")
            return
        
        # Run tests
        self.test_manager_init()
        self.test_emotion_mapping()
        self.test_runtime_switching()
        await self.test_automatic_integration()
        self.test_performance()
        self.test_fallback_behavior()
        self.test_config_persistence()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary"""
        console.print("\n" + "="*60)
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        
        table = Table(title="Test Results")
        table.add_column("Test", style="cyan")
        table.add_column("Result", style="bold")
        table.add_column("Details", style="dim")
        
        for result in self.results:
            status = "[green]PASS[/green]" if result["passed"] else "[red]FAIL[/red]"
            table.add_row(
                result["name"],
                status,
                result["details"]
            )
        
        console.print(table)
        
        # Overall result
        console.print(f"\n[bold]Overall: {passed}/{total} tests passed[/bold]")
        
        if passed == total:
            console.print("\n[bold green]✅ ALL TESTS PASSED![/bold green]")
            console.print("[dim]LoRA system is ready for production use[/dim]\n")
        else:
            console.print("\n[bold yellow]⚠️  SOME TESTS FAILED[/bold yellow]")
            console.print("[dim]Review failures and fix before deploying[/dim]\n")


async def main():
    """Main entry point"""
    tester = LoRASystemTester()
    
    try:
        await tester.run_all()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Tests cancelled[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]❌ Test suite error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    return


if __name__ == "__main__":
    asyncio.run(main())