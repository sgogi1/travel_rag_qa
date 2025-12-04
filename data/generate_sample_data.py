"""
Generate sample travel data for destinations and guides.
This creates realistic travel agency data for testing the RAG system.
Generates 100 destinations and 1000 guides.
"""

import json
import os
import random

# Base templates for generating diverse data
CITIES = [
    "Paris", "Lisbon", "Santorini", "Tokyo", "Bali", "Tuscany", "Iceland", 
    "Marrakech", "Kyoto", "Barcelona", "Rome", "London", "New York", "Dubai",
    "Singapore", "Sydney", "Cairo", "Bangkok", "Prague", "Vienna", "Amsterdam",
    "Berlin", "Madrid", "Seoul", "Hong Kong", "Mumbai", "Rio de Janeiro",
    "Buenos Aires", "Cape Town", "Lagos", "Nairobi", "Casablanca", "Istanbul",
    "Athens", "Stockholm", "Oslo", "Copenhagen", "Helsinki", "Dublin", "Edinburgh",
    "Zurich", "Geneva", "Brussels", "Warsaw", "Budapest", "Bucharest", "Sofia",
    "Belgrade", "Zagreb", "Ljubljana", "Reykjavik", "Luxembourg", "Monaco",
    "Valletta", "Nicosia", "Riga", "Tallinn", "Vilnius", "Minsk", "Kiev",
    "Moscow", "St. Petersburg", "Beijing", "Shanghai", "Guangzhou", "Shenzhen",
    "Taipei", "Manila", "Jakarta", "Kuala Lumpur", "Ho Chi Minh City", "Hanoi",
    "Phnom Penh", "Vientiane", "Yangon", "Dhaka", "Kathmandu", "Colombo",
    "Karachi", "Lahore", "Islamabad", "Kabul", "Tehran", "Baghdad", "Damascus",
    "Beirut", "Amman", "Jerusalem", "Tel Aviv", "Riyadh", "Jeddah", "Doha",
    "Kuwait City", "Manama", "Muscat", "Abu Dhabi", "Sharjah", "Almaty", "Tashkent"
]

COUNTRIES = [
    "France", "Portugal", "Greece", "Japan", "Indonesia", "Italy", "Iceland",
    "Morocco", "Spain", "United Kingdom", "United States", "UAE", "Singapore",
    "Australia", "Egypt", "Thailand", "Czech Republic", "Austria", "Netherlands",
    "Germany", "South Korea", "Hong Kong", "India", "Brazil", "Argentina",
    "South Africa", "Nigeria", "Kenya", "Turkey", "Sweden", "Norway", "Denmark",
    "Finland", "Ireland", "Switzerland", "Belgium", "Poland", "Hungary", "Romania",
    "Bulgaria", "Serbia", "Croatia", "Slovenia", "Luxembourg", "Monaco", "Malta",
    "Cyprus", "Latvia", "Estonia", "Lithuania", "Belarus", "Ukraine", "Russia",
    "China", "Taiwan", "Philippines", "Malaysia", "Vietnam", "Cambodia", "Laos",
    "Myanmar", "Bangladesh", "Nepal", "Sri Lanka", "Pakistan", "Afghanistan",
    "Iran", "Iraq", "Syria", "Lebanon", "Jordan", "Israel", "Saudi Arabia",
    "Qatar", "Kuwait", "Bahrain", "Oman", "Kazakhstan", "Uzbekistan"
]

ACTIVITIES_POOL = [
    "city tours", "museums", "wine tasting", "fine dining", "river cruises",
    "art galleries", "snorkeling", "water sports", "historical tours", "beaches",
    "seafood dining", "music venues", "sunset viewing", "photography tours",
    "culinary tours", "temple visits", "shopping", "cultural experiences",
    "sushi dining", "surfing", "yoga", "spa treatments", "cooking classes",
    "glacier tours", "hot springs", "whale watching", "Northern Lights viewing",
    "adventure tours", "desert tours", "hiking", "architecture tours", "nightlife",
    "tea ceremonies", "garden tours", "safari", "diving", "kayaking", "rafting",
    "paragliding", "skydiving", "bungee jumping", "rock climbing", "cycling",
    "horseback riding", "fishing", "bird watching", "wildlife viewing", "stargazing",
    "volcano tours", "cave exploration", "zip-lining", "helicopter tours",
    "balloon rides", "train journeys", "cruise tours", "island hopping",
    "market tours", "street food tours", "bar hopping", "jazz clubs",
    "opera", "theater", "festivals", "concerts", "sports events", "casinos",
    "beach volleyball", "tennis", "golf", "skiing", "snowboarding", "ice skating"
]

FIRST_NAMES = [
    "Maria", "Jean-Pierre", "Yuki", "Elena", "Marco", "Sarah", "Ahmad", "Kenji",
    "Carlos", "Putu", "Anna", "Giovanni", "Sophie", "Hans", "Mei", "Raj",
    "Fatima", "David", "Emma", "Luca", "Isabella", "Mohammed", "Chen", "Kim",
    "Ahmed", "Hassan", "Yusuf", "Ali", "Omar", "Zara", "Layla", "Noor", "Amira",
    "Sofia", "Elena", "Maya", "Aria", "Luna", "Zoe", "Chloe", "Olivia", "Emily",
    "James", "Michael", "William", "Daniel", "Matthew", "Christopher", "Andrew",
    "Joshua", "Joseph", "Thomas", "Ryan", "Nicholas", "Kevin", "Brian", "George",
    "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Steven", "Paul", "Mark",
    "Anthony", "Kenneth", "Stephen", "Donald", "Gary", "Eric", "Jacob", "Jonathan"
]

LAST_NAMES = [
    "Santos", "Dubois", "Tanaka", "Kostas", "Rossi", "Johnson", "Benali", "Yamamoto",
    "Mendez", "Sari", "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson",
    "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill",
    "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell",
    "Mitchell", "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz"
]

GUIDE_REGIONS = [
    "Lisbon, Portugal", "Paris, France", "Tokyo, Japan", "Santorini, Greece",
    "Tuscany, Italy", "Iceland", "Marrakech, Morocco", "Kyoto, Japan",
    "Barcelona, Spain", "Bali, Indonesia", "Rome, Italy", "London, UK",
    "New York, USA", "Dubai, UAE", "Singapore", "Sydney, Australia", "Cairo, Egypt",
    "Bangkok, Thailand", "Prague, Czech Republic", "Vienna, Austria",
    "Amsterdam, Netherlands", "Berlin, Germany", "Madrid, Spain", "Seoul, South Korea",
    "Hong Kong", "Mumbai, India", "Rio de Janeiro, Brazil", "Buenos Aires, Argentina",
    "Cape Town, South Africa", "Lagos, Nigeria", "Nairobi, Kenya", "Casablanca, Morocco",
    "Istanbul, Turkey", "Athens, Greece", "Stockholm, Sweden", "Oslo, Norway",
    "Copenhagen, Denmark", "Helsinki, Finland", "Dublin, Ireland", "Edinburgh, Scotland",
    "Zurich, Switzerland", "Geneva, Switzerland", "Brussels, Belgium", "Warsaw, Poland",
    "Budapest, Hungary", "Bucharest, Romania", "Sofia, Bulgaria", "Belgrade, Serbia",
    "Zagreb, Croatia", "Ljubljana, Slovenia", "Reykjavik, Iceland", "Valletta, Malta",
    "Nicosia, Cyprus", "Riga, Latvia", "Tallinn, Estonia", "Vilnius, Lithuania",
    "Moscow, Russia", "St. Petersburg, Russia", "Beijing, China", "Shanghai, China",
    "Taipei, Taiwan", "Manila, Philippines", "Jakarta, Indonesia", "Kuala Lumpur, Malaysia",
    "Ho Chi Minh City, Vietnam", "Hanoi, Vietnam", "Phnom Penh, Cambodia",
    "Vientiane, Laos", "Yangon, Myanmar", "Dhaka, Bangladesh", "Kathmandu, Nepal",
    "Colombo, Sri Lanka", "Karachi, Pakistan", "Lahore, Pakistan", "Islamabad, Pakistan",
    "Tehran, Iran", "Baghdad, Iraq", "Beirut, Lebanon", "Amman, Jordan",
    "Jerusalem, Israel", "Tel Aviv, Israel", "Riyadh, Saudi Arabia", "Jeddah, Saudi Arabia",
    "Doha, Qatar", "Kuwait City, Kuwait", "Manama, Bahrain", "Muscat, Oman",
    "Abu Dhabi, UAE", "Sharjah, UAE", "Almaty, Kazakhstan", "Tashkent, Uzbekistan"
]

DESCRIPTION_TEMPLATES_DEST = [
    "A vibrant {type} destination offering {features}. Experience {activities} and immerse yourself in {culture}. Perfect for {audience}.",
    "Discover {type} with stunning {features}. Enjoy {activities} and explore {culture}. Ideal for {audience}.",
    "Explore {type} featuring {features}. Participate in {activities} and discover {culture}. Great for {audience}.",
    "Visit {type} known for {features}. Engage in {activities} and experience {culture}. Perfect for {audience}.",
    "Experience {type} with {features}. Try {activities} and learn about {culture}. Ideal for {audience}."
]

DESCRIPTION_TEMPLATES_GUIDE = [
    "Local expert specializing in {specialty}. Fluent in {languages}. Offers {services}. Available for {tour_types}.",
    "Experienced guide with {experience} years of experience in {region}. Specializes in {specialty}. Provides {services}.",
    "Professional {type} guide offering {services}. Expert knowledge of {expertise}. Available for {tour_types}.",
    "Certified guide specializing in {specialty}. Fluent in {languages}. Offers personalized {services}.",
    "Local insider providing {services}. Deep knowledge of {expertise}. Specializes in {specialty}."
]


def generate_destination_description(name, country, activities):
    """Generate a description for a destination."""
    template = random.choice(DESCRIPTION_TEMPLATES_DEST)
    
    types = ["cosmopolitan city", "coastal town", "mountain retreat", "tropical paradise", 
             "historic city", "cultural hub", "adventure destination", "wellness retreat"]
    features = ["stunning architecture", "beautiful landscapes", "rich history", 
                "vibrant culture", "world-class cuisine", "breathtaking views"]
    cultures = ["local traditions", "authentic experiences", "cultural heritage", 
                "artistic scene", "culinary delights"]
    audiences = ["couples", "families", "solo travelers", "adventure seekers", 
                "culture enthusiasts", "food lovers", "photography enthusiasts"]
    
    return template.format(
        type=random.choice(types),
        features=random.choice(features),
        activities=", ".join(activities[:3]),
        culture=random.choice(cultures),
        audience=random.choice(audiences)
    )


def generate_guide_description(name, region, activities):
    """Generate a description for a guide."""
    template = random.choice(DESCRIPTION_TEMPLATES_GUIDE)
    
    specialties = ["historical tours", "culinary experiences", "cultural immersion",
                   "adventure activities", "photography tours", "wellness retreats"]
    languages = ["English", "Spanish", "French", "German", "Italian", "Portuguese",
                 "Japanese", "Chinese", "Arabic", "Russian"]
    services = ["personalized tours", "group experiences", "private excursions",
               "custom itineraries", "expert guidance"]
    tour_types = ["private tours", "group tours", "custom experiences"]
    types = ["cultural", "adventure", "culinary", "historical", "photography"]
    expertises = ["local history", "regional cuisine", "cultural traditions",
                 "hidden gems", "local insights"]
    
    return template.format(
        specialty=random.choice(specialties),
        languages=", ".join(random.sample(languages, random.randint(2, 4))),
        services=random.choice(services),
        tour_types=random.choice(tour_types),
        experience=random.randint(5, 20),
        region=region.split(",")[0] if "," in region else region,
        type=random.choice(types),
        expertise=random.choice(expertises)
    )


def generate_destinations(count=100):
    """Generate destination data."""
    destinations = []
    used_combinations = set()
    
    for i in range(count):
        # Ensure unique city-country combinations
        while True:
            city = random.choice(CITIES)
            country = random.choice(COUNTRIES)
            combo = (city, country)
            if combo not in used_combinations or len(used_combinations) > len(CITIES) * 2:
                used_combinations.add(combo)
                break
        
        # Generate activities (3-7 per destination)
        num_activities = random.randint(3, 7)
        activities = random.sample(ACTIVITIES_POOL, num_activities)
        
        description = generate_destination_description(city, country, activities)
        
        destinations.append({
            "name": city,
            "country": country,
            "description": description,
            "activities": activities
        })
    
    return destinations


def generate_guides(count=1000):
    """Generate guide data."""
    guides = []
    
    for i in range(count):
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        name = f"{first_name} {last_name}"
        
        region = random.choice(GUIDE_REGIONS)
        
        # Generate activities (2-5 per guide)
        num_activities = random.randint(2, 5)
        activities = random.sample(ACTIVITIES_POOL, num_activities)
        
        description = generate_guide_description(name, region, activities)
        
        guides.append({
            "name": name,
            "region": region,
            "description": description,
            "activities": activities
        })
    
    return guides


def generate_data():
    """Generate and save sample data files."""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Generating 100 destinations...")
    destinations = generate_destinations(100)
    
    print("Generating 1000 guides...")
    guides = generate_guides(1000)
    
    # Save destinations
    destinations_path = os.path.join(data_dir, "destinations.json")
    with open(destinations_path, 'w', encoding='utf-8') as f:
        json.dump(destinations, f, indent=2, ensure_ascii=False)
    print(f"✅ Generated {len(destinations)} destinations in {destinations_path}")
    
    # Save guides
    guides_path = os.path.join(data_dir, "guides.json")
    with open(guides_path, 'w', encoding='utf-8') as f:
        json.dump(guides, f, indent=2, ensure_ascii=False)
    print(f"✅ Generated {len(guides)} guides in {guides_path}")
    
    return destinations_path, guides_path


if __name__ == "__main__":
    generate_data()
