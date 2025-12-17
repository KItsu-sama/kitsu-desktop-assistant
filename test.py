


async def example_usage():
    """Example of how to use the integrated system"""
    
    # Create KitsuSelf (personality state)
    from core.personality.kitsu_self import KitsuSelf
    kitsu_self = KitsuSelf()
    
    # Create emotion engine with triggers
    engine = EmotionEngine(
        kitsu_self=kitsu_self,
        triggers_path="data/triggers.json"
    )
    
    # Link back to KitsuSelf
    kitsu_self.set_emotion_engine(engine)
    
    # Start background decay loop
    decay_task = asyncio.create_task(engine.run())
    
    # Simulate interactions
    print("=== Testing Emotion System ===\n")
    
    # Test 1: Praise
    print("User praises Kitsu...")
    engine.fire_trigger("praised")
    await asyncio.sleep(1)
    state = engine.get_state_dict()
    print(f"State: {state['mood']}/{state['style']} ({state['dominant_emotion']})")
    print(f"Avatar: {engine.get_avatar_hint()}\n")
    
    # Test 2: Insult
    print("User insults Kitsu...")
    engine.fire_trigger("insulted")
    await asyncio.sleep(1)
    state = engine.get_state_dict()
    print(f"State: {state['mood']}/{state['style']} ({state['dominant_emotion']})")
    print(f"Avatar: {engine.get_avatar_hint()}\n")
    
    # Test 3: Manual mood change
    print("Manually setting mood to flirty...")
    engine.set_mood("flirty")
    engine.set_style("sweet")
    state = engine.get_state_dict()
    print(f"State: {state['mood']}/{state['style']}")
    print(f"Avatar: {engine.get_avatar_hint()}\n")
    
    # Test 4: Hide
    print("Hiding Kitsu...")
    engine.hide()
    state = engine.get_state_dict()
    print(f"Hidden: {state['is_hidden']}")
    print(f"Mode: {state['current_mode']}\n")
    
    # Cleanup
    decay_task.cancel()


if __name__ == "__main__":
    asyncio.run(example_usage())