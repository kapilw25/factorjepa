"""
Convert YT_videos_raw.md to structured JSON, generate word frequency analysis and city matrix.
Re-run whenever raw.md is updated.

USAGE:
    python -u src/m00_data_prep.py --SANITY 2>&1 | tee logs/m00_data_prep_sanity.log
    python -u src/m00_data_prep.py --FULL 2>&1 | tee logs/m00_data_prep_full.log
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))

# Paths
RAW_MD = Path("Literature/Prev_work4/YT_videos_raw.md")
OUTPUT_JSON = Path("Literature/Prev_work4/YT_videos_raw.json")
OUTPUT_DIR = Path("outputs_data_prep")


def extract_youtube_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    if not url:
        return ""
    match = re.search(r'[?&]v=([a-zA-Z0-9_-]{11})', url)
    return match.group(1) if match else ""


def parse_video_entry(title: str, url: str) -> dict:
    """Create video entry dict with YouTube ID."""
    vid_id = extract_youtube_id(url)
    return {
        "id": vid_id,
        "title": title.strip(),
        "url": url.strip()
    }


def parse_markdown(md_content: str) -> dict:
    """Parse YT_videos_raw.md into structured data."""

    # City mappings
    city_map = {
        'A': 'delhi', 'B': 'mumbai', 'C': 'hyderabad',
        'D': 'bangalore', 'E': 'chennai', 'F': 'kolkata'
    }

    result = {
        "drive_tours": {c: [] for c in city_map.values()},
        "drone_views": {c: [] for c in city_map.values()},
        "walking_tours": {c: [] for c in city_map.values()},
        "tier2_cities": [],
        "monuments": [],
        "metadata": {}
    }
    result["walking_tours"]["goa"] = []

    lines = md_content.split('\n')
    current_section = None
    current_city = None
    current_monument = None
    current_tour_type = None
    i = 0

    tier2_cities = [
        "Jaipur", "Varanasi", "Lucknow", "Ahmedabad", "Pune", "Kochi",
        "Chandigarh", "Indore", "Bhopal", "Coimbatore", "Nagpur",
        "Visakhapatnam", "Surat", "Thiruvananthapuram", "Mysuru"
    ]

    monument_names = []

    while i < len(lines):
        line = lines[i].strip()

        # Detect main sections
        if line.startswith("## A. 4K Drive"):
            current_section = "drive"
            current_city = "delhi"
        elif line.startswith("## B. 4K Drive"):
            current_section = "drive"
            current_city = "mumbai"
        elif line.startswith("## C. 4K Drive"):
            current_section = "drive"
            current_city = "hyderabad"
        elif line.startswith("## D. 4K Drive"):
            current_section = "drive"
            current_city = "bangalore"
        elif line.startswith("## E. 4K Drive"):
            current_section = "drive"
            current_city = "chennai"
        elif line.startswith("## F. 4K Drive"):
            current_section = "drive"
            current_city = "kolkata"

        # Drone sections
        elif line.startswith("## A. 4K Drone"):
            current_section = "drone"
            current_city = "delhi"
        elif line.startswith("## B. 4K Drone"):
            current_section = "drone"
            current_city = "mumbai"
        elif line.startswith("## C. 4K Drone"):
            current_section = "drone"
            current_city = "hyderabad"
        elif line.startswith("## D. 4K Drone"):
            current_section = "drone"
            current_city = "bangalore"
        elif line.startswith("## E. 4K Drone"):
            current_section = "drone"
            current_city = "chennai"
        elif line.startswith("## F. 4K Drone"):
            current_section = "drone"
            current_city = "kolkata"

        # Walking tour sections
        elif line.startswith("## A. 4K Walking"):
            current_section = "walking"
            current_city = "delhi"
        elif line.startswith("## B. 4K Walking"):
            current_section = "walking"
            current_city = "mumbai"
        elif line.startswith("## C. 4K Walking"):
            current_section = "walking"
            current_city = "hyderabad"
        elif line.startswith("## D. 4K Walking"):
            current_section = "walking"
            current_city = "bangalore"
        elif line.startswith("## E. 4K Walking"):
            current_section = "walking"
            current_city = "chennai"
        elif line.startswith("## F. 4K Walking"):
            current_section = "walking"
            current_city = "kolkata"
        elif line.startswith("## G. 4K Walking") and "Goa" in line:
            current_section = "walking"
            current_city = "goa"

        # Tier 2 cities section
        elif line.startswith("## Tier 2 Cities"):
            current_section = "tier2"
            current_city = None

        # Monuments section (must come BEFORE tier2 city headers check)
        elif line.startswith("## Top 50 monuments"):
            current_section = "monuments"
            current_monument = None

        # Tier 2 city headers (only when in tier2 section)
        elif current_section == "tier2" and line.startswith("##"):
            for city in tier2_cities:
                if city.lower() in line.lower():
                    current_city = city.lower()
                    # Add placeholder if not exists
                    existing = [t for t in result["tier2_cities"] if t.get("city", "").lower() == current_city]
                    if not existing:
                        result["tier2_cities"].append({
                            "id": "",
                            "city": city,
                            "title": "",
                            "url": "",
                            "type": ""
                        })
                    break

        # Monument entry (- N. Name or - N. Name (City) :)
        elif current_section == "monuments" and re.match(r'^-\s*\d+\.', line):
            # Extract monument number and name
            match = re.match(r'^-\s*(\d+)\.\s*(.+?)(?:\s*:\s*)?$', line)
            if match:
                monument_num = int(match.group(1))
                raw_name = match.group(2).strip()
                # Extract city if in parentheses
                city_match = re.search(r'\(([^)]+)\)', raw_name)
                city = city_match.group(1) if city_match else ""
                # Remove city from name
                monument_name = re.sub(r'\s*\([^)]+\)\s*', '', raw_name).strip()

                current_monument = {
                    "id": monument_num,
                    "name": monument_name,
                    "city": city,
                    "walking_tours": [],
                    "drive_tours": [],
                    "drone_views": []
                }
                result["monuments"].append(current_monument)
                current_tour_type = None

        # Tour type under monument (##### 4K Walking Tour, ##### 4K Drive, ##### 4K Drone View)
        elif current_section == "monuments" and line.startswith("#####"):
            if "Walking" in line:
                current_tour_type = "walking_tours"
            elif "Drive" in line:
                current_tour_type = "drive_tours"
            elif "Drone" in line:
                current_tour_type = "drone_views"

        # Video entries - multiple formats:
        # 1. `- N. `title`` or `- N `title`` (drive/walk/drone sections)
        # 2. `- a. `title`` (monument sub-items)
        # 3. `` `title` `` on its own line (monument videos)

        # Check if line contains a backtick title
        title_match = re.search(r'`([^`]+)`', line)
        if title_match and not line.startswith("##"):
            title = title_match.group(1)
            # Look for URL on next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('http'):
                    url = next_line
                    entry = parse_video_entry(title, url)

                    # Add to appropriate section
                    if current_section == "drive" and current_city:
                        result["drive_tours"][current_city].append(entry)
                    elif current_section == "drone" and current_city:
                        result["drone_views"][current_city].append(entry)
                    elif current_section == "walking" and current_city:
                        result["walking_tours"][current_city].append(entry)
                    elif current_section == "monuments" and current_monument and current_tour_type:
                        current_monument[current_tour_type].append(entry)
                    elif current_section == "tier2" and current_city:
                        # Update tier2 entry
                        for t in result["tier2_cities"]:
                            if t.get("city", "").lower() == current_city:
                                t["id"] = extract_youtube_id(url)
                                t["title"] = title
                                t["url"] = url
                                break
                    i += 1  # Skip URL line

        i += 1

    # Calculate metadata
    drive_total = sum(len(v) for v in result["drive_tours"].values())
    drone_total = sum(len(v) for v in result["drone_views"].values())
    walking_total = sum(len(v) for v in result["walking_tours"].values())
    monument_videos = sum(
        len(m["walking_tours"]) + len(m["drive_tours"]) + len(m["drone_views"])
        for m in result["monuments"]
    )
    tier2_filled = sum(1 for t in result["tier2_cities"] if t.get("url"))

    result["metadata"] = {
        "total_drive_videos": drive_total,
        "total_drone_videos": drone_total,
        "total_walking_videos": walking_total,
        "total_monument_videos": monument_videos,
        "total_tier2_filled": tier2_filled,
        "total_tier2_blank": len(result["tier2_cities"]) - tier2_filled,
        "total_monuments": len(result["monuments"]),
        "grand_total": drive_total + drone_total + walking_total + monument_videos + tier2_filled
    }

    return result


def get_all_videos_with_context(data: dict) -> list:
    """Extract all videos with their section context for duplicate detection."""
    videos = []

    # Drive tours
    for city, vids in data.get("drive_tours", {}).items():
        for v in vids:
            if v.get("id"):
                videos.append((v["id"], f"drive/{city}", v.get("title", "")))

    # Drone views
    for city, vids in data.get("drone_views", {}).items():
        for v in vids:
            if v.get("id"):
                videos.append((v["id"], f"drone/{city}", v.get("title", "")))

    # Walking tours
    for city, vids in data.get("walking_tours", {}).items():
        for v in vids:
            if v.get("id"):
                videos.append((v["id"], f"walking/{city}", v.get("title", "")))

    # Tier2
    for v in data.get("tier2_cities", []):
        if v.get("id"):
            videos.append((v["id"], f"tier2/{v.get('city', '')}", v.get("title", "")))

    # Monuments
    for m in data.get("monuments", []):
        for tour_type in ["walking_tours", "drive_tours", "drone_views"]:
            for v in m.get(tour_type, []):
                if v.get("id"):
                    videos.append((v["id"], f"monument/{m.get('name', '')}", v.get("title", "")))

    return videos


def find_duplicate_ids(data: dict) -> dict:
    """Find duplicate YouTube IDs in the dataset."""
    videos = get_all_videos_with_context(data)
    id_counts = Counter([v[0] for v in videos])
    duplicates = {k: v for k, v in id_counts.items() if v > 1}

    # Get details for duplicates
    duplicate_details = {}
    for dup_id in duplicates:
        duplicate_details[dup_id] = [
            {"section": v[1], "title": v[2]}
            for v in videos if v[0] == dup_id
        ]

    return {
        "total_videos": len(videos),
        "unique_ids": len(id_counts),
        "duplicate_count": len(duplicates),
        "duplicates": duplicate_details
    }


def print_duplicates(dup_data: dict):
    """Print duplicate ID report."""
    print("\n" + "=" * 70)
    print("DUPLICATE YOUTUBE ID CHECK")
    print("=" * 70)
    print(f"Total videos:   {dup_data['total_videos']}")
    print(f"Unique IDs:     {dup_data['unique_ids']}")
    print(f"Duplicates:     {dup_data['duplicate_count']}")

    if dup_data['duplicate_count'] > 0:
        print("\n--- DUPLICATE ENTRIES ---")
        for dup_id, entries in dup_data['duplicates'].items():
            print(f"\n  {dup_id} appears {len(entries)} times:")
            for e in entries:
                print(f"    - [{e['section']}] {e['title'][:50]}...")
    else:
        print("\nNo duplicates found!")


def get_all_titles(data: dict) -> list:
    """Extract all video titles from parsed data."""
    titles = []

    # Drive tours
    for city, videos in data.get("drive_tours", {}).items():
        for v in videos:
            if v.get("title"):
                titles.append(v["title"])

    # Drone views
    for city, videos in data.get("drone_views", {}).items():
        for v in videos:
            if v.get("title"):
                titles.append(v["title"])

    # Walking tours
    for city, videos in data.get("walking_tours", {}).items():
        for v in videos:
            if v.get("title"):
                titles.append(v["title"])

    # Tier2
    for v in data.get("tier2_cities", []):
        if v.get("title"):
            titles.append(v["title"])

    # Monuments
    for m in data.get("monuments", []):
        for tour_type in ["walking_tours", "drive_tours", "drone_views"]:
            for v in m.get(tour_type, []):
                if v.get("title"):
                    titles.append(v["title"])

    return titles


def word_frequency_analysis(titles: list) -> dict:
    """Analyze word frequency and map to taxonomy categories."""

    # Taxonomy keywords
    taxonomy = {
        "time_of_day": ["morning", "afternoon", "evening", "night", "sunset", "sunrise", "dawn", "dusk", "golden hour"],
        "weather": ["rain", "rainy", "monsoon", "cloudy", "fog", "foggy", "clear", "sunny"],
        "crowd_density": ["busy", "crowded", "rush hour", "traffic", "peaceful", "quiet", "empty"],
        "scene_type": ["market", "beach", "temple", "fort", "palace", "heritage", "residential", "downtown", "flyover", "expressway", "highway"],
        "road_layout": ["narrow", "wide", "lane", "street", "road", "bridge", "tunnel", "coastal"],
        "notable_objects": ["bus", "metro", "train", "ferry", "bike", "auto", "rickshaw", "vendor", "shop"]
    }

    # Flatten all words
    all_words = []
    for title in titles:
        # Clean and tokenize
        words = re.findall(r'[a-zA-Z]+', title.lower())
        all_words.extend(words)

    word_counts = Counter(all_words)

    # Map to taxonomy
    taxonomy_counts = {cat: {} for cat in taxonomy}
    for category, keywords in taxonomy.items():
        for kw in keywords:
            kw_lower = kw.lower()
            # Check single word or phrase
            if ' ' in kw:
                count = sum(1 for t in titles if kw_lower in t.lower())
            else:
                count = word_counts.get(kw_lower, 0)
            if count > 0:
                taxonomy_counts[category][kw] = count

    return {
        "top_50_words": dict(word_counts.most_common(50)),
        "taxonomy_mapping": taxonomy_counts,
        "total_titles": len(titles),
        "total_words": len(all_words),
        "unique_words": len(word_counts)
    }


def create_city_matrix(data: dict) -> dict:
    """Create city x video_type matrix."""

    cities_tier1 = ["delhi", "mumbai", "hyderabad", "bangalore", "chennai", "kolkata"]
    cities_tier2 = ["jaipur", "varanasi", "lucknow", "ahmedabad", "pune", "kochi",
                    "chandigarh", "indore", "bhopal", "coimbatore", "nagpur",
                    "visakhapatnam", "surat", "thiruvananthapuram", "mysuru"]
    cities_other = ["goa"]

    matrix = {}

    # Tier 1 cities
    for city in cities_tier1:
        matrix[city] = {
            "tier": 1,
            "drive": len(data.get("drive_tours", {}).get(city, [])),
            "walking": len(data.get("walking_tours", {}).get(city, [])),
            "drone": len(data.get("drone_views", {}).get(city, []))
        }
        matrix[city]["total"] = matrix[city]["drive"] + matrix[city]["walking"] + matrix[city]["drone"]

    # Goa (special case - walking tours only in raw data)
    matrix["goa"] = {
        "tier": "other",
        "drive": len(data.get("drive_tours", {}).get("goa", [])),
        "walking": len(data.get("walking_tours", {}).get("goa", [])),
        "drone": len(data.get("drone_views", {}).get("goa", []))
    }
    matrix["goa"]["total"] = matrix["goa"]["drive"] + matrix["goa"]["walking"] + matrix["goa"]["drone"]

    # Tier 2 cities
    for city in cities_tier2:
        tier2_videos = [t for t in data.get("tier2_cities", []) if t.get("city", "").lower() == city]
        filled = sum(1 for t in tier2_videos if t.get("url"))
        matrix[city] = {
            "tier": 2,
            "drive": 0,
            "walking": 0,
            "drone": 0,
            "untagged": filled,
            "total": filled
        }

    # Summary
    tier1_total = sum(m["total"] for c, m in matrix.items() if m.get("tier") == 1)
    tier2_total = sum(m["total"] for c, m in matrix.items() if m.get("tier") == 2)
    other_total = sum(m["total"] for c, m in matrix.items() if m.get("tier") == "other")

    return {
        "matrix": matrix,
        "summary": {
            "tier1_total": tier1_total,
            "tier2_total": tier2_total,
            "other_total": other_total,
            "grand_total": tier1_total + tier2_total + other_total
        }
    }


def print_city_matrix(matrix_data: dict):
    """Print city matrix as formatted table."""
    matrix = matrix_data["matrix"]

    print("\n" + "=" * 70)
    print("CITY x VIDEO TYPE MATRIX")
    print("=" * 70)
    print(f"{'City':<20} {'Tier':<6} {'Drive':<8} {'Walk':<8} {'Drone':<8} {'Total':<8}")
    print("-" * 70)

    # Sort by tier then name
    tier1 = [(c, m) for c, m in matrix.items() if m.get("tier") == 1]
    tier2 = [(c, m) for c, m in matrix.items() if m.get("tier") == 2]
    other = [(c, m) for c, m in matrix.items() if m.get("tier") == "other"]

    print("--- TIER 1 ---")
    for city, m in sorted(tier1):
        print(f"{city.capitalize():<20} {m['tier']:<6} {m['drive']:<8} {m['walking']:<8} {m['drone']:<8} {m['total']:<8}")

    print("--- OTHER ---")
    for city, m in sorted(other):
        print(f"{city.capitalize():<20} {'--':<6} {m['drive']:<8} {m['walking']:<8} {m['drone']:<8} {m['total']:<8}")

    print("--- TIER 2 (placeholders) ---")
    for city, m in sorted(tier2):
        untagged = m.get('untagged', 0)
        status = f"({untagged} untagged)" if untagged else "(empty)"
        print(f"{city.capitalize():<20} {m['tier']:<6} {status}")

    print("-" * 70)
    s = matrix_data["summary"]
    print(f"{'TOTAL':<20} {'':<6} {s['tier1_total'] + s['other_total']:<8}")
    print(f"  Tier 1: {s['tier1_total']}, Other: {s['other_total']}, Tier 2: {s['tier2_total']}")


def print_word_frequency(freq_data: dict):
    """Print word frequency analysis."""
    print("\n" + "=" * 70)
    print("WORD FREQUENCY ANALYSIS")
    print("=" * 70)
    print(f"Total titles: {freq_data['total_titles']}")
    print(f"Total words: {freq_data['total_words']}")
    print(f"Unique words: {freq_data['unique_words']}")

    print("\n--- TOP 30 WORDS ---")
    for i, (word, count) in enumerate(list(freq_data['top_50_words'].items())[:30], 1):
        print(f"{i:2}. {word:<15} {count}")

    print("\n--- TAXONOMY MAPPING ---")
    for category, keywords in freq_data['taxonomy_mapping'].items():
        if keywords:
            print(f"\n{category.upper()}:")
            for kw, count in sorted(keywords.items(), key=lambda x: -x[1]):
                print(f"  {kw:<15} {count}")


def main():
    parser = argparse.ArgumentParser(description="Convert YT_videos_raw.md to JSON with analysis")
    parser.add_argument("--SANITY", action="store_true", help="Parse and show summary only")
    parser.add_argument("--FULL", action="store_true", help="Full conversion with all outputs")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    # Check input file
    if not RAW_MD.exists():
        print(f"ERROR: {RAW_MD} not found")
        sys.exit(1)

    print(f"=== Reading {RAW_MD} ===")
    md_content = RAW_MD.read_text(encoding='utf-8')
    print(f"File size: {len(md_content)} chars, {len(md_content.splitlines())} lines")

    # Task 1 & 2: Parse and convert to JSON
    print("\n=== Parsing markdown to JSON ===")
    data = parse_markdown(md_content)

    # Print metadata summary
    meta = data["metadata"]
    print(f"Drive videos:    {meta['total_drive_videos']}")
    print(f"Drone videos:    {meta['total_drone_videos']}")
    print(f"Walking videos:  {meta['total_walking_videos']}")
    print(f"Monument videos: {meta['total_monument_videos']}")
    print(f"Tier2 filled:    {meta['total_tier2_filled']}")
    print(f"Tier2 blank:     {meta['total_tier2_blank']}")
    print(f"Monuments:       {meta['total_monuments']}")
    print(f"---")
    print(f"GRAND TOTAL:     {meta['grand_total']} videos")

    # Task 3: Duplicate ID check
    print("\n=== Checking for duplicate YouTube IDs ===")
    dup_data = find_duplicate_ids(data)
    print(f"Checked {dup_data['total_videos']} videos, found {dup_data['duplicate_count']} duplicates")

    # Task 4: Word frequency
    print("\n=== Extracting titles for word frequency ===")
    titles = get_all_titles(data)
    print(f"Extracted {len(titles)} titles")
    freq_data = word_frequency_analysis(titles)

    # Task 5: City matrix
    print("\n=== Creating city matrix ===")
    matrix_data = create_city_matrix(data)

    if args.SANITY:
        print("\n--- SANITY MODE: Summary only ---")
        print_city_matrix(matrix_data)
        print("\nSANITY PASSED")

    elif args.FULL:
        # Save JSON
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {OUTPUT_JSON}")

        # Save word frequency
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        freq_file = OUTPUT_DIR / "word_frequency.json"
        with open(freq_file, 'w', encoding='utf-8') as f:
            json.dump(freq_data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {freq_file}")

        # Save matrix
        matrix_file = OUTPUT_DIR / "city_matrix.json"
        with open(matrix_file, 'w', encoding='utf-8') as f:
            json.dump(matrix_data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {matrix_file}")

        # Print detailed analysis
        print_duplicates(dup_data)
        print_city_matrix(matrix_data)
        print_word_frequency(freq_data)

        print(f"\nFULL COMPLETED")


if __name__ == "__main__":
    main()
