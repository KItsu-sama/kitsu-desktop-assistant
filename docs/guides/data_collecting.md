# Step 1: Initialize with samples
python scripts/quick_add.py --init-samples

# Step 2: Add your own facts
python scripts/quick_add.py "Cats are awesome" -c trivia
python scripts/quick_add.py "React is a JavaScript library" -c technical

# Step 3: Check what you have
python scripts/quick_add.py -l

# Step 4: Train overnight
python scripts/auto_pipeline.py

# Step 5: Wake up and chat!
python main.py