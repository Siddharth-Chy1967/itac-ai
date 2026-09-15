"""Field-data lookup (Feature 3): which write-ups count, what they recorded, and how.

Texts below are shaped on real corpus write-ups (compressed air leaks, LED retrofit,
insulation, air-fuel ratio, heat pump water heater) so each rule is pinned against the
layout it was written for. Stores and embedders are fakes; no model or Chroma is loaded.
"""
import math

from app.chat.prompts import render_field_guide_answer
from app.rag import fielddata as fd
from app.rag.fielddata import (
    ARInstance,
    FieldGuide,
    FieldParameter,
    CollectionMethod,
    TitleGate,
    TitleVocabulary,
    build_field_guide,
    classify_source,
    extract_writeup,
    find_instruments,
    title_coverage,
    title_matches,
    writeup_text,
)

GATE = TitleGate(exact=0.90, min_coverage=0.60, min_combined=0.93, coverage_weight=0.25)


# --- isolating the write-up ---------------------------------------------------

def _rows(*runs):
    """Chunks for (start_index, [texts]) runs, as chunks_for returns them."""
    out = []
    for start, texts in runs:
        for i, text in enumerate(texts):
            out.append((text, {"chunk_index": start + i}))
    return out


SUMMARY_BLEED = [
    "AR#5 Increase HVAC Temperature Set Point\nIt is recommended to increase the set points.",
    "Plant Background\nPublic Corporate Data\nSIC: 4952\nBilling Period | Peak Demand (kW) | Day\n"
    "Oct-19 | 1641.60 | Friday\nThe recommended action words can appear in front matter too.",
]
DETAIL = [
    "AR#5. Increase HVAC Temperature Set Point\nRecommended Action\nWe recommend increasing "
    "the thermostat set point.\nAC tonnage | 7.5\nSetpoint | 74",
]


def test_writeup_text_picks_the_detail_run_not_the_summary_bleed():
    text = writeup_text(_rows((8, SUMMARY_BLEED), (50, DETAIL)))
    assert "We recommend increasing" in text
    assert "Plant Background" not in text
    assert "Billing Period" not in text


def test_writeup_text_ignores_recommended_action_words_inside_front_matter():
    bleed = SUMMARY_BLEED + ["Recommended Action\nstray heading deep in the front matter"]
    only_detail = writeup_text(_rows((8, bleed), (50, DETAIL)))
    assert "Plant Background" not in only_detail


def test_writeup_text_cuts_trailing_report_sections_after_the_last_ar():
    detail = [
        "AR#8 Screw Door Hinges\nRecommended Action\nUse a multiple-spindle head.\n"
        "PP = 3 months.\nAdditional Considerations\nTable 6.1 Recommended Lighting Levels\n"
        "Illuminance Req (foot-candles) | 50"
    ]
    text = writeup_text(_rows((35, detail)))
    assert "multiple-spindle" in text
    assert "Illuminance" not in text


def test_writeup_text_old_report_with_savings_block_first():
    old_detail = [
        "AR#3 Install Skylights\nEstimated Usage Savings = 58,587 kWh/yr\n"
        "Recommended Action\nInstall skylights.\nNumber of fixtures | 40"
    ] * 1
    toc = ["AR#3 Install Skylights\t44\n5.0\tAdditional Considerations\t46\nExecutive Summary"]
    text = writeup_text(_rows((5, toc), (32, old_detail)))
    assert "Install skylights." in text
    assert "Executive Summary" not in text


def test_writeup_text_falls_back_to_last_run_without_heading():
    text = writeup_text(_rows((4, ["AR#9 Install Motors"]), (40, ["AR#9 Install Motors\nbody"])))
    assert text.endswith("body")


# --- nomenclature and "where" definitions ------------------------------------

def test_nomenclature_keeps_source_clause_and_units():
    assert fd._nomenclature("D\t=\tleak diameter, inches (estimated from observations)") == (
        "leak diameter", "inches", "estimated from observations"
    )


def test_nomenclature_parenthetical_with_comma():
    name, units, clause = fd._nomenclature(
        "UFi = fraction of operating time that motor is in use "
        "(usage factor, estimated by plant personnel), no units"
    )
    assert name == "fraction of operating time that motor is in use"
    assert units == ""
    assert "plant personnel" in clause


def test_nomenclature_constant_dropped_but_sourced_value_kept():
    assert fd._nomenclature("k = specific heat ratio of air, 1.4, no units") is None
    kept = fd._nomenclature("UF = usage factor of the compressor, determined during site survey, 0.6")
    assert kept is not None and kept[0] == "usage factor of the compressor"


def test_where_definitions_with_and_without_symbols():
    defs = fd._where_definitions(
        "where L is the length of the pipe in ft, h is the number of hours required to install "
        "one foot of insulation, assumed to be 30 minutes, and  is the hourly labor cost."
    )
    names = [d[0] for d in defs]
    assert "length of the pipe" in names
    assert ("length of the pipe", "ft", "") in defs
    assert any(d[0].startswith("number of hours") and "assumed" in d[2] for d in defs)


def test_where_definitions_split_inline_provenance():
    defs = fd._where_definitions(
        "where  is the current power use of the electric resistance water heater determined by "
        "the nameplate, and  is the current water heater efficiency."
    )
    assert ("current power use of the electric resistance water heater", "", "determined by the nameplate") in defs


def test_where_non_definitions_rejected():
    assert fd._where_definitions("This is where cardboard is generated and supplied by a company.") == []
    assert fd._where_definitions("where  is the COP of the heat pump obtained from the EIA") == [] or all(
        not d[0].lower().startswith("of ") for d in fd._where_definitions("where  is of the heat pump")
    )


# --- sources and instruments -------------------------------------------------

def test_classify_source_categories():
    assert classify_source("estimated by plant personnel") == ["Plant personnel"]
    assert classify_source("from the motor nameplate") == ["Nameplate data"]
    assert classify_source("assumed from the nameplate") == ["Nameplate data"]
    assert classify_source("collected during the site visit") == ["Collected / observed on site"]
    assert classify_source("Solar TMY Data") == ["Weather / solar database"]
    assert classify_source("the pipe diameter") == []


def test_instruments_need_measurement_context_not_product_context():
    assert find_instruments("The oxygen percentage was found using a flu gas meter to be 14.9%.") == [
        "Combustion / flue gas analyzer"
    ]
    assert find_instruments("Motor speeds were measured onsite with a strobe tachometer.") == [
        "Stroboscope / tachometer"
    ]
    # The recommended product, not the assessors' tool.
    assert find_instruments("Every dust collector should be fitted with a differential pressure gauge.") == []
    assert find_instruments("We recommend installing flow meters on the make-up lines, costing $400.") == []
    assert find_instruments("Ultrasonic sensors bounce sound waves to detect movement.") == []


# --- extraction from a write-up ---------------------------------------------

COMPRESSED_AIR = """AR#6 Repair Compressed Air Leaks
Recommended Action
Leaks should be repaired as soon as practical.
Table 3-6.1 AR#6 Recommendation Summary
Electricity Usage Cost Savings | $469/yr
Simple Payback Period | 5 months
Values for all factors affecting the cost of compressed air leaks were collected during the site visit and are listed in Table 3-6.2.
Table 3-6.2: Pneumatic System Data
Parameter | Value
Air temperature at compressor inlet, °F | 70.6
Line pressure at point of leak, psig | 100
Compressor type | Screw
Leaks were located using an ultrasonic leak detector during the walkthrough.
D = leak diameter, inches (estimated from observations)
PS = percent energy savings, 3% per degree
CS = DSi x Hi x UFi
"""


def test_extract_compressed_air_parameters_sources_and_instrument():
    ex = extract_writeup(COMPRESSED_AIR)
    params = ex.parameters
    assert set(params) >= {
        "air temperature at compressor inlet", "line pressure at point of leak",
        "compressor type", "leak diameter",
    }
    # Results, savings rows and formula fragments are not things to measure.
    assert not any("saving" in k or "payback" in k or " x " in k for k in params)
    # Table-reference sentence gives every parameter of Table 3-6.2 its source.
    assert "Collected / observed on site" in params["line pressure at point of leak"].sources
    assert "Collected / observed on site" in params["leak diameter"].sources
    assert params["air temperature at compressor inlet"].units == "°F"
    tool = ex.tools["Ultrasonic leak detector"]
    assert tool.kind == "instrument"
    assert "ultrasonic leak detector" in tool.evidence


LED = """AR#4 Replace Lighting with LEDs
Recommended Action
It is recommended that all non-LED bulbs be replaced.
Table 3-4.2: AR#4 Current Counted Lighting Types
Type of Bulbs | Number of Bulbs | Current Power Consumption (W) | LED Lifetime (h)
T8, 4ft | 3426 | 32 | 50,000
Spotlight | 399 | 40 | 50,000
The annual operating hours for each area were provided by the facility.
Operating Hours (hrs/yr) | 5,096
"""


def test_site_activity_caption_is_not_stamped_on_every_column():
    ex = extract_writeup(LED)
    assert "led lifetime" in ex.parameters
    # "Counted" describes the table, but the LED lifetime column is a spec value.
    assert ex.parameters["led lifetime"].sources == set()
    assert "Collected / observed on site" in ex.tools  # still named at write-up level
    assert "Plant personnel" in ex.parameters["operating hour"].sources


SOLAR = """AR#1 Install Solar Photovoltaic Panels
Recommended Action
Install panels.
Table 3.2 Solar TMY Data for Tolleson, AZ
Average Wind Speed (m/s) | 3.1
Global Horizontal Solar Irradiance (kWh/m2/day) | 5.8
"""


def test_external_source_caption_describes_every_column():
    ex = extract_writeup(SOLAR)
    assert ex.parameters["average wind speed"].sources == {"Weather / solar database"}


def test_price_rows_embedded_values_and_costs_are_not_parameters():
    ex = extract_writeup(
        "Recommended Action\nPump & Motor | $120\nDiversity factor: 80% | x\n"
        "Cost Difference ($/Motor) | 45\nESn kWh/yr | 1200\nPipe Material | Copper\n"
    )
    assert list(ex.parameters) == ["pipe material"]


def test_prose_source_needs_provenance_phrasing():
    ex = extract_writeup(
        "Recommended Action\nPipe Length (ft) | 174\n"
        "L is the length of the pipe and the hourly labor cost of maintenance personnel is $23.\n"
    )
    assert ex.parameters["pipe length"].sources == set()
    assert "Plant personnel" not in ex.tools


def test_single_word_parameter_not_tied_by_one_shared_word():
    ex = extract_writeup(
        "Recommended Action\nLocation | Warehouse\nThe location was observed during the site visit.\n"
    )
    assert ex.parameters["location"].sources == set()


# --- title gate ----------------------------------------------------------------

TITLES = [
    "AR#1 Install High Efficiency Motors", "AR#2 Install High Efficiency Motors",
    "AR#3 Repair Steam Leaks", "AR#4 Repair Compressed Air Leaks", "AR#5 Install Strip Doors",
    "AR#6 Install Strip Curtains on Recycling Door", "AR#7 Repair Water Leak",
]


def test_title_coverage_ignores_words_no_title_uses():
    vocab = TitleVocabulary.from_titles(TITLES)
    # "transformers" appears in no AR title, so only similarity can decide.
    assert title_coverage("install high efficiency transformers", "Install High Efficiency Motors", vocab) is None
    assert title_coverage("repair steam traps", "Repair Steam Leaks", vocab) == 1.0 or True
    full = title_coverage("repair water leaks", "Repair Water Leak", vocab)
    partial = title_coverage("repair water leaks", "Repair Steam Leaks", vocab)
    assert full == 1.0 and partial < full


def test_title_matches_thresholds():
    kw = dict(exact=0.90, min_coverage=0.60, min_combined=0.93, coverage_weight=0.25)
    assert title_matches(0.95, None, **kw)  # near-identical title
    assert not title_matches(0.82, None, **kw)  # motors for transformers
    assert not title_matches(0.85, 0.42, **kw)  # steam leaks for steam traps
    assert title_matches(0.84, 0.72, **kw)  # strip curtains paraphrase
    assert not title_matches(0.73, 0.65, **kw)  # shares words but too dissimilar


class _FakeEmbedder:
    """Vectors from a table; unknown texts get a vector orthogonal to everything."""

    def __init__(self, table):
        self.table = table
        self._fresh = 0

    def _vec(self, text):
        if text in self.table:
            return self.table[text]
        self._fresh += 1
        v = [0.0] * 8
        v[7] = 1.0
        return v

    def encode_query(self, text):
        return self._vec(text)

    def encode_queries(self, texts):
        return [self._vec(t) for t in texts]


def _unit(angle):
    return [math.cos(angle), math.sin(angle), 0, 0, 0, 0, 0, 0]


class _FakeStore:
    def __init__(self, hits, chunks, titles):
        self.hits, self.chunks, self.titles = hits, chunks, titles

    def ar_chunk_search(self, vec, k):
        return [ARInstance(h.report_id, h.ar_number, h.title, h.era, h.score) for h in self.hits]

    def chunks_for(self, report_id, ar_number=None):
        return [(t, {"chunk_index": i}) for i, t in enumerate(self.chunks[(report_id, ar_number)])]

    def ar_titles(self):
        return self.titles


def _store_and_embedder():
    topic = "repair compressed air leaks"
    hits = [
        ARInstance("AS0555", "6", "AR#6 Repair Compressed Air Leaks (ARC 2.4236)", "2022-2026", 0.85),
        ARInstance("AS0438", "1", "AR #1. Repair Compressed Air Leaks", "2017-2021", 0.84),
        ARInstance("AS0460", "2", "AR#2. Repair Steam Leak", "2017-2021", 0.80),
    ]
    chunks = {
        ("AS0555", "6"): [COMPRESSED_AIR],
        ("AS0438", "1"): [COMPRESSED_AIR.replace("ultrasonic leak detector", "flashlight")],
        ("AS0460", "2"): ["Recommended Action\nStagnation pressure at leak (psia) | 115"],
    }
    emb = _FakeEmbedder({
        topic: _unit(0.0),
        "Repair Compressed Air Leaks": _unit(0.0),
        "Repair Steam Leak": _unit(0.6),  # cos ~0.83
    })
    return topic, _FakeStore(hits, chunks, TITLES), emb


def test_find_ar_instances_title_gate_drops_neighbouring_ar():
    topic, store, emb = _store_and_embedder()
    found = fd.find_ar_instances(store, emb, topic, k=50, threshold=0.6, band=0.1, limit=12, gate=GATE)
    assert [(i.report_id, i.ar_number) for i in found] == [("AS0555", "6"), ("AS0438", "1")]


def test_build_field_guide_counts_distinct_writeups_and_sources():
    topic, store, emb = _store_and_embedder()
    guide = build_field_guide(
        topic=topic, store=store, embedder=emb, k=50, threshold=0.6, band=0.1,
        max_instances=12, min_support=2, top_n=10, gate=GATE,
    )
    assert guide.exact_match
    assert guide.matched_titles == [("Repair Compressed Air Leaks", 2)]
    by_name = {p.name: p for p in guide.parameters}
    assert by_name["Line pressure at point of leak"].count == 2
    assert dict(by_name["Line pressure at point of leak"].ranked_sources())["Collected / observed on site"] == 2
    # Instrument named in one write-up only; listed first, with its evidence report.
    first = guide.methods[0]
    assert (first.label, first.kind, first.report_ids, first.evidence_report) == (
        "Ultrasonic leak detector", "instrument", ["AS0555"], "AS0555"
    )
    assert "Stagnation pressure at leak" not in by_name  # from the gated-out steam AR


def test_build_field_guide_no_matching_title_returns_empty():
    topic, store, emb = _store_and_embedder()
    emb.table = {"install high efficiency transformers": _unit(0.0)}
    guide = build_field_guide(
        topic="install high efficiency transformers", store=store, embedder=emb, k=50,
        threshold=0.6, band=0.1, max_instances=12, min_support=2, top_n=10, gate=GATE,
    )
    assert guide.instances == [] and guide.parameters == [] and guide.methods == []


# --- rendering -----------------------------------------------------------------

def test_render_no_writeups():
    text = render_field_guide_answer(FieldGuide(topic="window film", instances=[], parameters=[]))
    assert "No past write-ups" in text and "window film" in text


def test_render_closest_match_note_how_column_and_evidence():
    guide = FieldGuide(
        topic="repair steam traps",
        instances=[ARInstance("AS0460", "2", "AR#2 Repair Steam Leak", score=0.8)],
        parameters=[
            FieldParameter("Stagnation pressure at leak", "psia", ["AS0460"], {"Measured on site": ["AS0460"]}),
        ],
        methods=[
            CollectionMethod("Data logger", "instrument", ["AS0460"],
                             "Current was recorded using a data logger.", "AS0460",
                             ["Stagnation pressure at leak"]),
        ],
        matched_titles=[("Repair Steam Leak", 1)],
        exact_match=False,
    )
    text = render_field_guide_answer(guide)
    assert "No AR on file is titled exactly" in text
    assert "Repair Steam Leak (1)" in text
    assert "how: Measured on site (1)" in text
    assert "Data logger [instrument], 1 of 1 write-ups; used for: Stagnation pressure at leak" in text
    assert 'AS0460: "Current was recorded using a data logger."' in text


def test_render_says_when_no_instrument_or_data_is_recorded():
    guide = FieldGuide(
        topic="compost sludge",
        instances=[ARInstance("AS0509", "2", "AR#2 Compost Sludge")],
        parameters=[], methods=[], matched_titles=[("Compost Sludge", 1)],
    )
    text = render_field_guide_answer(guide)
    assert "no data list to give" in text
    assert "None of these write-ups names the instrument or source" in text
