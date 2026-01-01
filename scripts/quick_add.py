# ============================================================================
# FILE: scripts/quick_add.py
# Quick command-line tool for adding facts to knowledge base
# ============================================================================

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()

# ============================================================================
# Quick Add Tool
# ============================================================================

class QuickAdd:
    """Command-line tool for quickly adding facts to knowledge base"""
    
    def __init__(self):
        self.knowledge_dir = Path("data/knowledge")
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        self.categories = {
            "facts": "facts.txt",
            "trivia": "trivia.txt", 
            "technical": "technical.txt",
            "science": "science.txt",
            "history": "history.txt",
            "culture": "culture.txt",
            "language": "language.txt",
            "gaming": "gaming.txt"
        }
    
    def add_fact(self, text: str, category: str = "facts"):
        """Add a single fact"""
        if category not in self.categories:
            console.print(f"[red]Unknown category: {category}[/red]")
            console.print(f"Valid: {', '.join(self.categories.keys())}")
            return False
        
        file_path = self.knowledge_dir / self.categories[category]
        
        # Ensure newline at end
        if not text.endswith('\n'):
            text += '\n'
        
        # Append
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(text)
        
        console.print(f"[green]✅ Added to {category}:[/green] {text.strip()}")
        return True
    
    def add_batch(self, facts: list, category: str = "facts"):
        """Add multiple facts at once"""
        count = 0
        for fact in facts:
            if self.add_fact(fact, category):
                count += 1
        
        console.print(f"\n[bold green]✅ Added {count} facts to {category}[/bold green]")
    
    def interactive_add(self):
        """Interactive mode"""
        console.print("\n[bold magenta]🦊 KITSU KNOWLEDGE BASE - QUICK ADD[/bold magenta]\n")
        
        # Show categories
        table = Table(title="Available Categories")
        table.add_column("Category", style="cyan")
        table.add_column("File", style="yellow")
        table.add_column("Auto-Tags", style="green")
        
        tags = {
            "facts": "neutral/behave/chaotic",
            "trivia": "playful/behave/chaotic",
            "technical": "helpful/behave/sweet",
            "science": "curious/behave/sweet",
            "history": "interested/behave/chaotic",
            "culture": "interested/behave/sweet",
            "language": "helpful/behave/chaotic",
            "gaming": "playful/behave/chaotic"
        }
        
        for cat, file in self.categories.items():
            table.add_row(cat, file, tags.get(cat, "neutral"))
        
        console.print(table)
        console.print()
        
        # Get category
        category = Prompt.ask(
            "Category",
            choices=list(self.categories.keys()),
            default="facts"
        )
        
        console.print(f"\n[cyan]Adding to:[/cyan] {category}")
        console.print("[dim]Enter facts (one per line, empty line to finish):[/dim]\n")
        
        facts = []
        while True:
            try:
                fact = input("  > ")
                if not fact.strip():
                    break
                facts.append(fact.strip())
            except (EOFError, KeyboardInterrupt):
                break
        
        if facts:
            self.add_batch(facts, category)
        else:
            console.print("[yellow]No facts entered[/yellow]")
    
    def list_files(self):
        """List all knowledge files and their line counts"""
        console.print("\n[bold cyan]📚 Knowledge Base Files[/bold cyan]\n")
        
        table = Table()
        table.add_column("File", style="cyan")
        table.add_column("Lines", style="yellow", justify="right")
        table.add_column("Size", style="green", justify="right")
        
        total_lines = 0
        total_size = 0
        
        for file_name in self.categories.values():
            file_path = self.knowledge_dir / file_name
            
            if file_path.exists():
                lines = len(file_path.read_text(encoding='utf-8').splitlines())
                size = file_path.stat().st_size
                
                total_lines += lines
                total_size += size
                
                table.add_row(
                    file_name,
                    str(lines),
                    f"{size / 1024:.1f} KB"
                )
            else:
                table.add_row(file_name, "0", "0 KB", style="dim")
        
        console.print(table)
        console.print(f"\n[bold]Total:[/bold] {total_lines} facts, {total_size / 1024:.1f} KB\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quick add facts to Kitsu's knowledge base"
    )
    
    parser.add_argument(
        "text",
        nargs="*",
        help="Fact text (or use -i for interactive)"
    )
    
    parser.add_argument(
        "-c", "--category",
        default="facts",
        choices=["facts", "trivia", "technical", "science", "history", "culture", "language", "gaming"],
        help="Category (default: facts)"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode"
    )
    
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all knowledge files"
    )
    
    parser.add_argument(
        "-f", "--file",
        type=Path,
        help="Import facts from file (one per line)"
    )
    
    parser.add_argument(
        "--init-samples",
        action="store_true",
        help="Initialize with 200 sample facts"
    )
    
    args = parser.parse_args()
    
    tool = QuickAdd()
    
    # List mode
    if args.list:
        tool.list_files()
        return 0
    
    # Initialize samples
    if args.init_samples:
        init_sample_facts()
        return 0
    
    # Interactive mode
    if args.interactive:
        tool.interactive_add()
        return 0
    
    # File import
    if args.file:
        if not args.file.exists():
            console.print(f"[red]File not found: {args.file}[/red]")
            return 1
        
        facts = [
            line.strip() 
            for line in args.file.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        
        tool.add_batch(facts, args.category)
        return 0
    
    # Single fact from args
    if args.text:
        fact = " ".join(args.text)
        tool.add_fact(fact, args.category)
        return 0
    
    # No args - show help
    parser.print_help()
    return 0


# ============================================================================
# 200 Sample Facts for Kitsu
# ============================================================================

def init_sample_facts():
    """Initialize knowledge base with 200 sample facts"""
    
    console.print("\n[bold magenta]🦊 Initializing Kitsu's Knowledge Base[/bold magenta]\n")
    console.print("[cyan]Adding 200 sample facts...[/cyan]\n")
    
    tool = QuickAdd()
    
    # General Facts (50)
    general_facts = [
        "The Earth is approximately 4.54 billion years old.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The human body has 206 bones.",
        "Light travels at 299,792,458 meters per second.",
        "The Amazon Rainforest produces 20% of Earth's oxygen.",
        "Honey never spoils and can last thousands of years.",
        "Octopuses have three hearts and blue blood.",
        "The Great Wall of China is over 13,000 miles long.",
        "Venus is the hottest planet in our solar system.",
        "Dolphins sleep with one eye open.",
        "A year on Mercury is only 88 Earth days.",
        "Bananas are berries but strawberries aren't.",
        "The Pacific Ocean covers more than 30% of Earth's surface.",
        "Mount Everest grows about 4 millimeters every year.",
        "Sharks have been around longer than trees.",
        "The human brain uses 20% of the body's energy.",
        "A bolt of lightning is five times hotter than the sun.",
        "The Sahara Desert is roughly the size of the United States.",
        "Polar bears are left-handed.",
        "The moon is moving away from Earth at 3.8 cm per year.",
        "Cows have best friends and get stressed when separated.",
        "The shortest war lasted 38 minutes between Britain and Zanzibar.",
        "Your nose can remember 50,000 different scents.",
        "Gold is edible and used in some fancy foods.",
        "A day on Venus is longer than a year on Venus.",
        "The Eiffel Tower can be 15 cm taller in summer due to heat.",
        "Butterflies taste with their feet.",
        "The longest recorded flight of a chicken is 13 seconds.",
        "Hot water freezes faster than cold water (Mpemba effect).",
        "Humans and giraffes have the same number of neck bones.",
        "The inventor of the Pringles can is buried in one.",
        "A cloud can weigh more than a million pounds.",
        "Rabbits can't vomit.",
        "The unicorn is Scotland's national animal.",
        "A sneeze travels at about 100 miles per hour.",
        "Wombat poop is cube-shaped.",
        "The shortest complete sentence in English is 'I am.'",
        "Apples float because they are 25% air.",
        "A group of flamingos is called a flamboyance.",
        "The tongue is the strongest muscle in the body.",
        "Penguins mate for life.",
        "A jiffy is an actual unit of time (1/100th of a second).",
        "The world's oldest known recipe is for beer.",
        "Tigers have striped skin, not just striped fur.",
        "A single strand of spaghetti is called a spaghetto.",
        "The inventor of the frisbee was turned into a frisbee after death.",
        "Sea otters hold hands while sleeping so they don't drift apart.",
        "A group of pugs is called a grumble.",
        "The smell of freshly cut grass is a plant distress signal.",
        "Alaska is the only state that can be typed on one row of a keyboard."
    ]
    
    # Trivia (40)
    trivia_facts = [
        "The shortest war in history was between Britain and Zanzibar on August 27, 1896, lasting 38 minutes.",
        "A group of crows is called a murder.",
        "The longest English word without a vowel is 'rhythms'.",
        "Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid.",
        "A shrimp's heart is in its head.",
        "Nintendo was founded in 1889 as a playing card company.",
        "The word 'nerd' was first coined by Dr. Seuss in 'If I Ran the Zoo'.",
        "Oxford University is older than the Aztec Empire.",
        "The inventor of the microwave oven only received $2 for his discovery.",
        "A single cloud can weigh more than a million pounds.",
        "The fear of long words is called hippopotomonstrosesquippedaliophobia.",
        "Bubble wrap was originally invented as wallpaper.",
        "The shortest complete sentence is 'Go.'",
        "A group of owls is called a parliament.",
        "The Twitter bird's name is Larry.",
        "Nutmeg is extremely poisonous if injected intravenously.",
        "The first computer bug was an actual bug found in a Harvard computer in 1947.",
        "A group of hedgehogs is called a prickle.",
        "The dot over the letter 'i' is called a tittle.",
        "A jiffy is an actual unit of time: 1/100th of a second.",
        "Scotland's national animal is the unicorn.",
        "Astronauts can't burp in space.",
        "A day on Venus is longer than its year.",
        "The first oranges weren't orange - they were green.",
        "Bananas are curved because they grow towards the sun.",
        "A group of pandas is called an embarrassment.",
        "The hashtag symbol is called an octothorpe.",
        "A crocodile cannot stick its tongue out.",
        "Sea otters have the densest fur in the animal kingdom.",
        "Honey never spoils; archaeologists found 3000-year-old honey in Egyptian tombs that was still edible.",
        "A bolt of lightning contains enough energy to toast 100,000 slices of bread.",
        "The longest recorded flight of a chicken is 13 seconds.",
        "A single strand of spaghetti is called a spaghetto.",
        "The smell of freshly cut grass is actually a plant distress call.",
        "Octopuses have three hearts and blue blood.",
        "Wombat poop is cube-shaped.",
        "A group of flamingos is called a flamboyance.",
        "The inventor of the Pringles can is buried in one.",
        "A sneeze travels at about 100 mph.",
        "Apples float in water because they are 25% air."
    ]
    
    # Technical/Programming (40)
    technical_facts = [
        "Python was named after Monty Python, not the snake.",
        "The first computer programmer was Ada Lovelace in the 1840s.",
        "The first computer virus was created in 1983 and was called 'Elk Cloner'.",
        "CAPTCHA stands for 'Completely Automated Public Turing test to tell Computers and Humans Apart'.",
        "The first electronic computer weighed more than 27 tons.",
        "WiFi stands for Wireless Fidelity.",
        "The '@' symbol is called an 'at sign' or 'commercial at'.",
        "The first domain name ever registered was Symbolics.com on March 15, 1985.",
        "Email existed before the World Wide Web.",
        "The first mouse was made of wood.",
        "Linux is named after its creator, Linus Torvalds.",
        "The term 'bug' in computing comes from an actual moth found in a computer in 1947.",
        "JavaScript was created in just 10 days.",
        "The first webcam was created to monitor a coffee pot at Cambridge University.",
        "USB stands for Universal Serial Bus.",
        "The first YouTube video was uploaded on April 23, 2005, titled 'Me at the zoo'.",
        "HTML stands for HyperText Markup Language.",
        "The first iPhone was released on June 29, 2007.",
        "Git was created by Linus Torvalds, the same person who created Linux.",
        "The programming language 'C' was named after 'B', which was named after 'BCPL'.",
        "The first computer game was created in 1962 and called 'Spacewar!'.",
        "RAM stands for Random Access Memory.",
        "The first version of Windows was released in 1985.",
        "JPEG stands for Joint Photographic Experts Group.",
        "The term 'cookie' in web browsing comes from 'magic cookie', a Unix term.",
        "The first tweet was sent by Jack Dorsey on March 21, 2006.",
        "Bluetooth is named after a Danish king, Harald Bluetooth.",
        "PDF stands for Portable Document Format.",
        "The first computer hard drive weighed over a ton and stored 5MB.",
        "Python's philosophy is documented in 'The Zen of Python'.",
        "The Windows 95 startup sound was composed by Brian Eno.",
        "The first Apple computer sold for $666.66.",
        "Moore's Law predicts that computing power doubles every two years.",
        "The term 'spam' for junk email comes from a Monty Python sketch.",
        "The first computer mouse had only one button.",
        "CSS stands for Cascading Style Sheets.",
        "The programming language Java was originally called Oak.",
        "The first laptop computer weighed 24 pounds.",
        "URL stands for Uniform Resource Locator.",
        "The @ symbol in email was chosen by Ray Tomlinson in 1971."
    ]
    
    # Science (40)
    science_facts = [
        "DNA was first discovered in 1869 by Friedrich Miescher.",
        "The human body contains about 37.2 trillion cells.",
        "Atoms are 99.9999999% empty space.",
        "A teaspoon of neutron star would weigh 6 billion tons.",
        "The hottest temperature ever recorded was 5.5 trillion degrees Celsius.",
        "Sound travels through water 4 times faster than through air.",
        "The human eye can distinguish about 10 million different colors.",
        "Antarctica is the driest place on Earth, not the Sahara.",
        "A single bolt of lightning contains enough energy to cook 100,000 pieces of toast.",
        "The universe is approximately 13.8 billion years old.",
        "Protons have a lifetime of at least 10^34 years.",
        "The speed of light is exactly 299,792,458 meters per second.",
        "Water can exist in three states simultaneously at the triple point.",
        "Graphene is 200 times stronger than steel.",
        "The coldest temperature ever recorded was -273.15°C (absolute zero was nearly reached).",
        "Time slows down the faster you move, according to Einstein's relativity.",
        "Quantum entanglement allows particles to affect each other instantly across any distance.",
        "The human brain generates about 20 watts of power.",
        "Glass is technically a liquid, not a solid.",
        "Helium is the only element that cannot be solidified by cooling alone.",
        "A photon takes 40,000 years to travel from the Sun's core to its surface.",
        "The Earth's core is as hot as the surface of the Sun.",
        "Water is the only substance that expands when it freezes.",
        "The human body glows in the dark, but the light is too weak to see.",
        "Sharks existed before trees evolved.",
        "The average human body contains enough carbon to make 900 pencils.",
        "Bananas are radioactive due to their potassium-40 content.",
        "The coldest place in the universe is on Earth (in a lab).",
        "Mercury is the only metal that is liquid at room temperature.",
        "The Mariana Trench is deeper than Mount Everest is tall.",
        "A single sneeze can travel up to 100 mph.",
        "The human stomach produces a new layer of mucus every two weeks.",
        "Diamond and graphite are both made of pure carbon.",
        "Light from the Sun takes 8 minutes and 20 seconds to reach Earth.",
        "The human body is made up of about 60% water.",
        "Neutron stars spin at about 600 rotations per second.",
        "The observable universe is about 93 billion light-years in diameter.",
        "Black holes are regions where gravity is so strong that nothing can escape.",
        "Antimatter exists but is extremely rare and expensive to produce.",
        "The placebo effect can work even when you know it's a placebo."
    ]
    
    # History (30)
    history_facts = [
        "The ancient Egyptians worshipped over 2,000 gods and goddesses.",
        "The shortest reign of a monarch was Louis XIX of France, who ruled for 20 minutes.",
        "The Roman Empire lasted for about 1,000 years.",
        "The Great Wall of China was built over several dynasties spanning 2,000 years.",
        "The ancient city of Pompeii was destroyed by Mount Vesuvius in 79 AD.",
        "The Rosetta Stone was the key to deciphering Egyptian hieroglyphics.",
        "The first written constitution was created in 1787 in the United States.",
        "The ancient Olympics were held for over 1,000 years before being banned.",
        "The Hundred Years' War actually lasted 116 years.",
        "The guillotine was used in France until 1977.",
        "The first emperor of China, Qin Shi Huang, was buried with over 8,000 terracotta warriors.",
        "The Renaissance began in Italy in the 14th century.",
        "The printing press was invented by Johannes Gutenberg in 1440.",
        "The first modern democracy was established in ancient Athens.",
        "The Magna Carta was signed in 1215, limiting the power of the English king.",
        "The Age of Exploration began in the 15th century.",
        "The French Revolution began in 1789 with the storming of the Bastille.",
        "The Industrial Revolution started in Britain in the late 18th century.",
        "The first airplane flight by the Wright Brothers lasted only 12 seconds.",
        "World War I began in 1914 and ended in 1918.",
        "The Titanic sank on its maiden voyage in 1912.",
        "The first human in space was Yuri Gagarin in 1961.",
        "The Berlin Wall fell in 1989, ending the Cold War era.",
        "The first successful vaccine was developed for smallpox in 1796.",
        "The ancient Library of Alexandria was one of the largest libraries in the ancient world.",
        "The Viking Age lasted from approximately 793 to 1066 AD.",
        "The Black Death killed an estimated 75-200 million people in the 14th century.",
        "The Silk Road was an ancient network of trade routes connecting East and West.",
        "The first moon landing occurred on July 20, 1969.",
        "The ancient Mayan calendar was incredibly accurate, predicting solar eclipses centuries in advance."
    ]
    
    # Add all facts
    tool.add_batch(general_facts, "facts")
    tool.add_batch(trivia_facts, "trivia")
    tool.add_batch(technical_facts, "technical")
    tool.add_batch(science_facts, "science")
    tool.add_batch(history_facts, "history")
    
    console.print("\n[bold green]✅ Initialized with 200 sample facts![/bold green]")
    console.print("\n[cyan]📊 Summary:[/cyan]")
    console.print(f"   Facts: {len(general_facts)}")
    console.print(f"   Trivia: {len(trivia_facts)}")
    console.print(f"   Technical: {len(technical_facts)}")
    console.print(f"   Science: {len(science_facts)}")
    console.print(f"   History: {len(history_facts)}")
    console.print(f"   [bold]Total: 200 facts[/bold]")
    
    console.print("\n[cyan]🎯 Next steps:[/cyan]")
    console.print("   1. Add more facts: [yellow]python scripts/quick_add.py -i[/yellow]")
    console.print("   2. Run training: [yellow]python scripts/Auto_pipeline.py[/yellow]")
    console.print("\n[dim]Kitsu will learn all these facts! 🦊[/dim]")


if __name__ == "__main__":
    sys.exit(main())