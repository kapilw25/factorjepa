# WalkIndia-200K: Next Steps

*Date: Tue Feb 3, 2026*

---

## 1. Follow Detailed Taxonomy
   - Use Professor's **India World Model Taxonomy** for stratification + slice reporting (NOT supervision)

## 2. Data Collection: 200K Clips from 700+ Videos
   - **2.1** Remove YouTube channel branding
   - **2.2** Scene detect: min 4s, max 10s
   - **2.3** Cities: Delhi, Goa, Hyderabad, Mumbai
   - **2.4** Monuments (50): [1. Gateway of India, 2. India Gate, 3. Charminar, 4. Taj Mahal, 5. Qutub Minar, 6. Red Fort, 7. Hawa Mahal, 8. Amer Fort, 9. Jantar Mantar, 10. City Palace Jaipur, 11. Mehrangarh Fort, 12. Jaisalmer Fort, 13. Chittorgarh Fort, 14. Ranthambore Fort, 15. Junagarh Fort, 16. Fatehpur Sikri, 17. Agra Fort, 18. Sanchi Stupa, 19. Khajuraho, 20. Gwalior Fort, 21. Konark Sun Temple, 22. Jagannath Temple, 23. Lingaraja Temple, 24. Mahabodhi Temple, 25. Nalanda Mahavihara, 26. Victoria Memorial, 27. Howrah Bridge, 28. Somnath Temple, 29. Rani ki Vav, 30. Statue of Unity, 31. Sabarmati Ashram, 32. Golconda Fort, 33. Meenakshi Amman Temple, 34. Brihadeeswara Temple, 35. Shore Temple, 36. Ramanathaswamy Temple, 37. Mysore Palace, 38. Hampi, 39. Virupaksha Temple Hampi, 40. Gol Gumbaz, 41. Ajanta Caves, 42. Ellora Caves, 43. Elephanta Caves, 44. Siddhivinayak Temple, 45. Shaniwar Wada, 46. Golden Temple, 47. Jallianwala Bagh, 48. Rock Garden, 49. Lotus Temple, 50. Humayun's Tomb]

## 3. HuggingFace Dataset
   - Make dataset private
   - Create datacard with Video viewer

## 4. Factor-JEPA Implementation
   > Ref: `@Literature/Prev_work4/proposal_FactorJEPA.pdf`

   - **4.1** Section 9: Evaluating V-JEPA on Indian Urban Walking Clips
   - **4.2** Section 10: Continual Self-Supervised Pretraining on Indian Urban Clips
   - **4.3** Section 11: Surgery Fine-Tuning - Progressive Prefix Unfreezing with Factor Datasets

---

## 5. India World Model Taxonomy

> **Purpose**: Stratification + slice reporting ONLY (NOT supervision)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INDIA WORLD MODEL TAXONOMY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐   ┌──────────────┐   ┌─────────────┐   ┌───────────┐ │
│  │ scene_type       │   │ time_of_day  │   │ weather     │   │ crowd     │ │
│  │ [single-select]  │   │ [single]     │   │ [single]    │   │ [single]  │ │
│  ├──────────────────┤   ├──────────────┤   ├─────────────┤   ├───────────┤ │
│  │ • market         │   │ • morning    │   │ • clear     │   │ • low     │ │
│  │ • junction       │   │ • afternoon  │   │ • rain      │   │ • med     │ │
│  │ • residential    │   │ • evening    │   │ • fog       │   │ • high    │ │
│  │ • promenade      │   │ • night      │   └─────────────┘   └───────────┘ │
│  │ • transit        │   └──────────────┘                                   │
│  │ • temple_tourist │                                                      │
│  │ • highway        │   ┌─────────────────────┐   ┌─────────────────────┐  │
│  │ • alley          │   │ notable_objects     │   │ road_layout         │  │
│  └──────────────────┘   │ [multi-select]      │   │ [multi-select]      │  │
│                         ├─────────────────────┤   ├─────────────────────┤  │
│                         │ • bus               │   │ • intersection      │  │
│                         │ • auto_rickshaw     │   │ • narrow_lane       │  │
│                         │ • bike              │   │ • wide_road         │  │
│                         │ • street_vendor     │   │ • sidewalk_present  │  │
│                         │ • police            │   │ • median            │  │
│                         │ • signage           │   └─────────────────────┘  │
│                         │ • animals           │                            │
│                         └─────────────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Category | Type | Options |
|----------|------|---------|
| scene_type | single | market, junction, residential, promenade, transit, temple_tourist, highway, alley |
| time_of_day | single | morning, afternoon, evening, night |
| weather | single | clear, rain, fog |
| crowd_density | single | low, med, high |
| notable_objects | multi | bus, auto_rickshaw, bike, street_vendor, police, signage, animals |
| road_layout | multi | intersection, narrow_lane, wide_road, sidewalk_present, median |

---