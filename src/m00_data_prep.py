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
OUTPUT_JSON = Path("src/utils/YT_videos_raw.json")
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

    tier2_city_names = [
        "jaipur", "varanasi", "lucknow", "ahmedabad", "pune", "kochi",
        "chandigarh", "indore", "bhopal", "coimbatore", "nagpur",
        "visakhapatnam", "surat", "thiruvananthapuram", "mysuru"
    ]

    result = {
        "drive_tours": {c: [] for c in city_map.values()},
        "drone_views": {c: [] for c in city_map.values()},
        "walking_tours": {c: [] for c in city_map.values()},
        "tier2_cities": {c: {"drive": [], "walking": [], "drone": [], "rain": []} for c in tier2_city_names},
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

        # Tier 2 city headers (## N: CityName or ## N. CityName)
        elif current_section == "tier2" and line.startswith("##") and not line.startswith("#####"):
            for city in tier2_cities:
                if city.lower() in line.lower():
                    current_city = city.lower()
                    current_tour_type = None  # Reset tour type when entering new city
                    break

        # Tier 2 tour type headers (##### X.Y: CityName : 4K Walking/Drive/Drone/Rain)
        elif current_section == "tier2" and line.startswith("#####"):
            line_lower = line.lower()
            if "walking" in line_lower or "walk" in line_lower:
                current_tour_type = "walking"
            elif "drive" in line_lower or "ride" in line_lower:
                current_tour_type = "drive"
            elif "drone" in line_lower:
                current_tour_type = "drone"
            elif "rain" in line_lower:
                current_tour_type = "rain"

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
                        # Add video to tier2 city's appropriate category
                        if current_city in result["tier2_cities"]:
                            tour_type = current_tour_type if current_tour_type else "walking"
                            result["tier2_cities"][current_city][tour_type].append(entry)
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

    # Tier 2 totals (now a dict of cities with drive/walking/drone/rain lists)
    tier2_drive = sum(len(city_data["drive"]) for city_data in result["tier2_cities"].values())
    tier2_walking = sum(len(city_data["walking"]) for city_data in result["tier2_cities"].values())
    tier2_drone = sum(len(city_data["drone"]) for city_data in result["tier2_cities"].values())
    tier2_rain = sum(len(city_data["rain"]) for city_data in result["tier2_cities"].values())
    tier2_total = tier2_drive + tier2_walking + tier2_drone + tier2_rain
    tier2_cities_with_videos = sum(1 for city_data in result["tier2_cities"].values()
                                    if any(len(city_data[t]) > 0 for t in ["drive", "walking", "drone", "rain"]))

    result["metadata"] = {
        "total_drive_videos": drive_total,
        "total_drone_videos": drone_total,
        "total_walking_videos": walking_total,
        "total_monument_videos": monument_videos,
        "total_tier2_videos": tier2_total,
        "total_tier2_drive": tier2_drive,
        "total_tier2_walking": tier2_walking,
        "total_tier2_drone": tier2_drone,
        "total_tier2_rain": tier2_rain,
        "tier2_cities_filled": tier2_cities_with_videos,
        "tier2_cities_empty": len(result["tier2_cities"]) - tier2_cities_with_videos,
        "total_monuments": len(result["monuments"]),
        "grand_total": drive_total + drone_total + walking_total + monument_videos + tier2_total
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

    # Tier2 (now dict of cities with drive/walking/drone/rain lists)
    for city, city_data in data.get("tier2_cities", {}).items():
        for tour_type in ["drive", "walking", "drone", "rain"]:
            for v in city_data.get(tour_type, []):
                if v.get("id"):
                    videos.append((v["id"], f"tier2/{city}/{tour_type}", v.get("title", "")))

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

    # Tier2 (dict of cities with drive/walking/drone/rain lists)
    for city, city_data in data.get("tier2_cities", {}).items():
        for tour_type in ["drive", "walking", "drone", "rain"]:
            for v in city_data.get(tour_type, []):
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

    # Tier 2 cities (now dict structure with drive/walking/drone/rain)
    tier2_data = data.get("tier2_cities", {})
    for city in cities_tier2:
        city_data = tier2_data.get(city, {})
        drive_count = len(city_data.get("drive", []))
        walking_count = len(city_data.get("walking", []))
        drone_count = len(city_data.get("drone", []))
        rain_count = len(city_data.get("rain", []))
        matrix[city] = {
            "tier": 2,
            "drive": drive_count,
            "walking": walking_count,
            "drone": drone_count,
            "rain": rain_count,
            "total": drive_count + walking_count + drone_count + rain_count
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


def print_summary_tables(data: dict, matrix_data: dict):
    """Print 4 summary tables for all videos."""
    matrix = matrix_data["matrix"]

    # Calculate monument totals
    monument_drive = sum(len(m.get("drive_tours", [])) for m in data.get("monuments", []))
    monument_walk = sum(len(m.get("walking_tours", [])) for m in data.get("monuments", []))
    monument_drone = sum(len(m.get("drone_views", [])) for m in data.get("monuments", []))
    monument_total = monument_drive + monument_walk + monument_drone

    # Tier 1 totals
    tier1_cities = ["delhi", "mumbai", "hyderabad", "bangalore", "chennai", "kolkata"]
    t1_drive = sum(matrix[c]["drive"] for c in tier1_cities)
    t1_walk = sum(matrix[c]["walking"] for c in tier1_cities)
    t1_drone = sum(matrix[c]["drone"] for c in tier1_cities)
    t1_total = t1_drive + t1_walk + t1_drone

    # Tier 2 totals (now with drive/walking/drone/rain)
    tier2_cities = ["jaipur", "varanasi", "lucknow", "ahmedabad", "pune", "kochi",
                    "chandigarh", "indore", "bhopal", "coimbatore", "nagpur",
                    "visakhapatnam", "surat", "thiruvananthapuram", "mysuru"]
    t2_drive = sum(matrix[c].get("drive", 0) for c in tier2_cities)
    t2_walk = sum(matrix[c].get("walking", 0) for c in tier2_cities)
    t2_drone = sum(matrix[c].get("drone", 0) for c in tier2_cities)
    t2_rain = sum(matrix[c].get("rain", 0) for c in tier2_cities)
    t2_total = t2_drive + t2_walk + t2_drone + t2_rain

    # Goa
    goa_drive = matrix["goa"]["drive"]
    goa_walk = matrix["goa"]["walking"]
    goa_drone = matrix["goa"]["drone"]
    goa_total = goa_drive + goa_walk + goa_drone

    # Grand totals
    grand_drive = t1_drive + goa_drive + monument_drive + t2_drive
    grand_walk = t1_walk + goa_walk + monument_walk + t2_walk
    grand_drone = t1_drone + goa_drone + monument_drone + t2_drone
    grand_rain = t2_rain
    grand_total = t1_total + goa_total + monument_total + t2_total

    # ===== TABLE 1: Overall Summary =====
    print("\n" + "=" * 80)
    print("TABLE 1: OVERALL SUMMARY (All Videos)")
    print("=" * 80)
    print(f"{'Category':<25} {'Drive':>8} {'Walk':>8} {'Drone':>8} {'Rain':>8} {'Total':>8}")
    print("-" * 80)
    print(f"{'Tier 1 (6 cities)':<25} {t1_drive:>8} {t1_walk:>8} {t1_drone:>8} {0:>8} {t1_total:>8}")
    print(f"{'Goa':<25} {goa_drive:>8} {goa_walk:>8} {goa_drone:>8} {0:>8} {goa_total:>8}")
    print(f"{'Tier 2 (15 cities)':<25} {t2_drive:>8} {t2_walk:>8} {t2_drone:>8} {t2_rain:>8} {t2_total:>8}")
    print(f"{'Monuments':<25} {monument_drive:>8} {monument_walk:>8} {monument_drone:>8} {0:>8} {monument_total:>8}")
    print("-" * 80)
    print(f"{'GRAND TOTAL':<25} {grand_drive:>8} {grand_walk:>8} {grand_drone:>8} {grand_rain:>8} {grand_total:>8}")

    # ===== TABLE 2: Tier 1 Cities =====
    print("\n" + "=" * 70)
    print("TABLE 2: TIER 1 CITIES")
    print("=" * 70)
    print(f"{'City':<20} {'Drive':>8} {'Walk':>8} {'Drone':>8} {'Total':>8}")
    print("-" * 70)
    # Sort by total descending
    tier1_sorted = sorted(tier1_cities, key=lambda c: matrix[c]["total"], reverse=True)
    for city in tier1_sorted:
        m = matrix[city]
        print(f"{city.capitalize():<20} {m['drive']:>8} {m['walking']:>8} {m['drone']:>8} {m['total']:>8}")
    print("-" * 70)
    print(f"{'TOTAL':<20} {t1_drive:>8} {t1_walk:>8} {t1_drone:>8} {t1_total:>8}")

    # ===== TABLE 3: Tier 2 + Goa =====
    print("\n" + "=" * 80)
    print("TABLE 3: TIER 2 CITIES + GOA")
    print("=" * 80)
    print(f"{'City':<20} {'Drive':>8} {'Walk':>8} {'Drone':>8} {'Rain':>8} {'Total':>8}")
    print("-" * 80)
    print(f"{'Goa':<20} {goa_drive:>8} {goa_walk:>8} {goa_drone:>8} {0:>8} {goa_total:>8}")
    print("-" * 80)
    # Tier 2 cities with videos first, sorted by total descending
    tier2_filled = [(c, matrix[c]["total"]) for c in tier2_cities if matrix[c]["total"] > 0]
    tier2_empty_count = sum(1 for c in tier2_cities if matrix[c]["total"] == 0)

    for city, _ in sorted(tier2_filled, key=lambda x: -x[1]):
        m = matrix[city]
        print(f"{city.capitalize():<20} {m['drive']:>8} {m['walking']:>8} {m['drone']:>8} {m['rain']:>8} {m['total']:>8}")

    if tier2_empty_count > 0:
        print(f"{'(' + str(tier2_empty_count) + ' cities empty)':<20} {0:>8} {0:>8} {0:>8} {0:>8} {0:>8}")
    print("-" * 80)
    print(f"{'TOTAL':<20} {goa_drive + t2_drive:>8} {goa_walk + t2_walk:>8} {goa_drone + t2_drone:>8} {t2_rain:>8} {goa_total + t2_total:>8}")

    # ===== TABLE 4: Monuments =====
    print("\n" + "=" * 70)
    print(f"TABLE 4: MONUMENTS ({monument_total} videos / {len(data.get('monuments', []))} listed)")
    print("=" * 70)
    print(f"{'Monument':<30} {'City':<15} {'Drive':>6} {'Walk':>6} {'Drone':>6} {'Total':>6}")
    print("-" * 70)

    monuments_with_videos = []
    monuments_empty = 0
    for m in data.get("monuments", []):
        drive = len(m.get("drive_tours", []))
        walk = len(m.get("walking_tours", []))
        drone = len(m.get("drone_views", []))
        total = drive + walk + drone
        if total > 0:
            monuments_with_videos.append((m["name"], m.get("city", ""), drive, walk, drone, total))
        else:
            monuments_empty += 1

    for name, city, drive, walk, drone, total in sorted(monuments_with_videos, key=lambda x: -x[5]):
        print(f"{name[:28]:<30} {city[:13]:<15} {drive:>6} {walk:>6} {drone:>6} {total:>6}")

    if monuments_empty > 0:
        print(f"{'(' + str(monuments_empty) + ' monuments empty)':<30} {'-':<15} {0:>6} {0:>6} {0:>6} {0:>6}")
    print("-" * 70)
    print(f"{'TOTAL':<30} {'':<15} {monument_drive:>6} {monument_walk:>6} {monument_drone:>6} {monument_total:>6}")


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
    print(f"Tier1 Drive:     {meta['total_drive_videos']}")
    print(f"Tier1 Drone:     {meta['total_drone_videos']}")
    print(f"Tier1 Walking:   {meta['total_walking_videos']}")
    print(f"Tier2 videos:    {meta['total_tier2_videos']} (drive={meta['total_tier2_drive']}, walk={meta['total_tier2_walking']}, drone={meta['total_tier2_drone']}, rain={meta['total_tier2_rain']})")
    print(f"Tier2 cities:    {meta['tier2_cities_filled']} filled, {meta['tier2_cities_empty']} empty")
    print(f"Monument videos: {meta['total_monument_videos']}")
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
        print_summary_tables(data, matrix_data)
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
        print_summary_tables(data, matrix_data)
        print_word_frequency(freq_data)

        print(f"\nFULL COMPLETED")


if __name__ == "__main__":
    main()
