"""
Generates a realistic training dataset for fake news detection.
Modeled after the WELFake dataset structure (label: 0=real, 1=fake).
Features are drawn from known linguistic patterns in fake vs real news research.
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ---------- Real news patterns ----------
real_templates = [
    "The {org} released a report on {topic} showing {stat} increase over the past {period}.",
    "Officials from {country} confirmed {n} casualties following {event} in {region}.",
    "{company} reported quarterly earnings of ${amount} billion, {dir} analyst expectations.",
    "A new study published in {journal} links {cause} to {effect} among {demographic}.",
    "The Federal Reserve raised interest rates by {bps} basis points citing {reason}.",
    "{country}'s parliament passed legislation on {topic} with {n} votes in favor.",
    "Health authorities warned of a rise in {disease} cases across {region} this {season}.",
    "Scientists at {university} have identified a new {discovery} in {field}.",
    "The United Nations called for {action} amid growing concerns over {crisis}.",
    "Stock markets fell {pct}% after data showed {indicator} slowing unexpectedly.",
    "Researchers confirmed that {drug} reduces {condition} risk by {pct}% in clinical trials.",
    "{city} mayor announced a ${amount} million infrastructure plan for {year}.",
    "The World Health Organization updated its guidelines on {topic} after new evidence emerged.",
    "A federal judge ruled against {entity} in a landmark {topic} case on {day}.",
    "{country} and {country2} signed a bilateral trade agreement covering {sector}.",
    "The {agency} released data showing unemployment fell to {pct}% last month.",
    "Scientists detected traces of {substance} in {location}, raising environmental concerns.",
    "Tech giant {company} announced layoffs affecting {n} employees across {n2} offices.",
    "Central bank minutes revealed concerns about {topic} in the next fiscal quarter.",
    "Experts say {policy} could reduce carbon emissions by {pct}% by {year}.",
]

real_slots = {
    "org": ["WHO", "IMF", "World Bank", "CDC", "IPCC", "NASA", "OECD"],
    "topic": ["climate change", "inflation", "cybersecurity", "public health", "immigration policy", "energy transition"],
    "stat": ["a 3.2%", "an 8%", "a 12%", "a 5.7%"],
    "period": ["fiscal year", "quarter", "decade", "six months"],
    "country": ["Germany", "Japan", "Brazil", "South Korea", "Canada", "Australia"],
    "country2": ["France", "India", "Mexico", "Indonesia", "UK", "Nigeria"],
    "n": ["12", "47", "230", "3", "18"],
    "n2": ["14", "5", "32"],
    "event": ["the earthquake", "the flood", "the industrial accident", "the conflict"],
    "region": ["the northern region", "coastal areas", "the capital district", "southeastern provinces"],
    "company": ["Apple", "Toyota", "HSBC", "Shell", "Siemens", "Samsung"],
    "amount": ["2.3", "14.7", "0.8", "5.1", "320"],
    "dir": ["beating", "missing", "matching"],
    "journal": ["The Lancet", "Nature", "NEJM", "Science", "JAMA"],
    "cause": ["air pollution", "sedentary lifestyles", "processed food consumption", "sleep deprivation"],
    "effect": ["cardiovascular disease", "cognitive decline", "metabolic syndrome"],
    "demographic": ["adults over 50", "urban populations", "children under 12", "adolescents"],
    "bps": ["25", "50", "75"],
    "reason": ["persistent inflation", "labor market tightness", "economic overheating"],
    "discovery": ["compound", "species", "mechanism", "gene variant", "protein structure"],
    "field": ["immunology", "particle physics", "marine biology", "neuroscience"],
    "action": ["humanitarian aid", "immediate ceasefire", "sanctions review", "diplomatic talks"],
    "crisis": ["food insecurity", "the refugee situation", "climate impacts", "debt levels"],
    "pct": ["3.2", "7.5", "12.4", "0.8", "4.1", "18"],
    "indicator": ["consumer spending", "manufacturing output", "export growth", "retail sales"],
    "drug": ["the new vaccine", "the treatment", "the therapy", "the compound"],
    "condition": ["heart disease", "stroke", "diabetes", "infection"],
    "city": ["New York", "London", "Paris", "Tokyo", "Sydney", "Berlin"],
    "year": ["2025", "2026", "2030"],
    "entity": ["the corporation", "the state government", "the defendant", "the agency"],
    "day": ["Tuesday", "Wednesday", "Friday"],
    "sector": ["agricultural goods", "technology products", "financial services", "clean energy"],
    "agency": ["BLS", "ONS", "Eurostat", "Statistics Canada"],
    "substance": ["microplastics", "PFAS chemicals", "heavy metals", "nitrates"],
    "location": ["groundwater samples", "soil near industrial zones", "river sediments"],
    "policy": ["the carbon tax", "stricter vehicle emissions standards", "renewable energy subsidies"],
    "university": ["MIT", "Oxford", "ETH Zurich", "Stanford", "Tokyo University"],
    "disease": ["dengue fever", "measles", "respiratory infections", "gastrointestinal illness"],
    "season": ["summer", "winter", "spring"],
}

# ---------- Fake news patterns ----------
fake_templates = [
    "{celebrity} SECRETLY {action} and the mainstream media is HIDING it!",
    "BREAKING: {govt} is planning to {conspiracy} by {method} — whistleblower reveals ALL",
    "Doctors WON'T tell you this: {substance} CURES {disease} in just {time}!",
    "EXPOSED: {company} has been {wrongdoing} for YEARS and nobody is talking about it",
    "Scientists ADMIT {fact} was a LIE all along — here's what THEY don't want you to know",
    "URGENT: {product} found in {food} is {harm} — share before they DELETE this",
    "The {govt} is using {tech} to {conspiracy2} every citizen — LEAKED documents prove it",
    "SHOCKING VIDEO: {politician} caught {scandal} — career OVER?",
    "New law will make it ILLEGAL to {activity} starting {date} — are you prepared?",
    "{country} preparing to INVADE {country2} as {event2} escalates, sources say",
    "WARNING: {vaccine} contains {ingredient} that causes {side_effect} in {pct}% of patients",
    "DEEP STATE plot to {conspiracy3} revealed by {source} — media blackout in effect",
    "They're putting {chemical} in {food2} to make you {effect2} — here's the PROOF",
    "BILLIONAIRE admits funding {conspiracy4} to control {resource} — watch before removed",
    "{celebrity2} is actually {claim} — shocking new photos PROVE it",
    "BANNED RESEARCH: {topic2} found to cause {effect3} — study SUPPRESSED by {org2}",
    "Government FORCING {mandate} on all citizens by {date2} — fight back NOW",
    "{institution} COLLAPSES tonight? Insiders reveal the TRUTH they're hiding from you",
    "MIRACLE: Man cures {disease2} in {time2} using only {remedy} — Big Pharma is FURIOUS",
    "EXPOSED: Elections in {country3} were rigged using {method2} — the evidence is UNDENIABLE",
]

fake_slots = {
    "celebrity": ["A-list Hollywood star", "Famous pop singer", "Major sports icon", "Top TV host"],
    "celebrity2": ["A famous actor", "A major celebrity", "A well-known musician"],
    "action": ["converted to a secret religion", "fled the country", "faked their own death"],
    "govt": ["The government", "The deep state", "The global elite", "World leaders"],
    "conspiracy": ["microchip all citizens", "collapse the economy", "ban cash", "control the food supply"],
    "conspiracy2": ["monitor", "track", "spy on", "control"],
    "conspiracy3": ["overthrow democracy", "enslave the population", "control the food supply"],
    "conspiracy4": ["the pandemic", "the climate agenda", "the financial system"],
    "method": ["vaccines", "chemtrails", "5G towers", "the water supply", "fluoride"],
    "method2": ["hacked voting machines", "fake ballots", "AI manipulation", "rigged algorithms"],
    "substance": ["this household spice", "this common herb", "one simple fruit", "this cheap mineral"],
    "disease": ["cancer", "diabetes", "COVID-19", "Alzheimer's", "heart disease"],
    "disease2": ["stage 4 cancer", "diabetes", "arthritis", "chronic fatigue"],
    "time": ["3 days", "one week", "48 hours", "two weeks"],
    "time2": ["30 days", "2 weeks", "10 days"],
    "company": ["Big Pharma", "A major tech giant", "A top social media company"],
    "wrongdoing": ["poisoning the water", "suppressing cures", "bribing officials", "hiding data"],
    "fact": ["climate change", "the moon landing", "vaccine safety", "germ theory"],
    "product": ["a common household item", "a popular food additive", "a widely used chemical"],
    "food": ["tap water", "popular cereals", "baby formula", "processed meats"],
    "food2": ["the water supply", "popular snack foods", "fast food", "canned goods"],
    "harm": ["linked to cancer", "destroying your brain", "causing infertility", "poisoning you slowly"],
    "tech": ["5G towers", "AI surveillance", "quantum satellites", "smart meters"],
    "politician": ["a top senator", "a world leader", "a senior official"],
    "scandal": ["accepting bribes", "in a secret meeting", "destroying documents", "lying under oath"],
    "activity": ["own guns", "grow your own food", "use cash", "homeschool your children"],
    "date": ["January 1st", "next month", "this fall", "March 15th"],
    "date2": ["January", "next year", "this summer"],
    "country": ["Russia", "China", "The US", "Iran", "North Korea"],
    "country2": ["Ukraine", "Taiwan", "Israel", "Poland", "South Korea"],
    "country3": ["a major country", "the US", "Brazil", "Germany"],
    "event2": ["the border crisis", "the proxy war", "diplomatic tensions", "the conflict"],
    "vaccine": ["The mRNA vaccine", "This new shot", "The flu vaccine", "The booster"],
    "ingredient": ["graphene oxide", "live parasites", "nanobots", "mercury compounds"],
    "side_effect": ["permanent brain damage", "DNA alteration", "sterilization", "microchip activation"],
    "pct": ["73", "47", "91", "63"],
    "source": ["an anonymous insider", "a brave whistleblower", "leaked Pentagon files"],
    "chemical": ["mind-control chemicals", "fertility-reducing agents", "cancer-causing additives"],
    "effect2": ["compliant and docile", "addicted and dependent", "infertile", "easier to control"],
    "resource": ["the world's oil", "global food production", "freshwater supplies", "gold reserves"],
    "claim": ["a robot", "a government agent", "using a body double", "secretly deceased"],
    "topic2": ["5G radiation", "GMO food", "fluoride in water", "artificial sweeteners"],
    "effect3": ["permanent memory loss", "sterility", "cancer within 5 years", "autism"],
    "org2": ["Big Pharma", "the WHO", "government regulators", "the FDA"],
    "mandate": ["mandatory vaccines", "digital ID", "social credit scores", "GPS tracking"],
    "institution": ["the banking system", "the stock market", "the US dollar", "social media"],
    "remedy": ["lemon juice and baking soda", "essential oils", "raw garlic", "apple cider vinegar"],
}

def fill_template(template, slots):
    import re
    keys = re.findall(r'\{(\w+)\}', template)
    result = template
    for key in keys:
        if key in slots:
            result = result.replace('{' + key + '}', random.choice(slots[key]), 1)
    return result

def generate_samples(templates, slots, n, label):
    rows = []
    for _ in range(n):
        tmpl = random.choice(templates)
        text = fill_template(tmpl, slots)
        rows.append({'text': text, 'label': label})
    return rows

# Additional hand-crafted real examples
extra_real = [
    {"text": "The European Central Bank held interest rates steady amid mixed signals on inflation recovery.", "label": 0},
    {"text": "A peer-reviewed study in Nature found that urban heat islands increased city temperatures by 2°C on average.", "label": 0},
    {"text": "Canada announced $500 million in aid for wildfire recovery efforts in western provinces.", "label": 0},
    {"text": "The International Monetary Fund revised its global growth forecast downward to 2.8% for the year.", "label": 0},
    {"text": "Researchers at Johns Hopkins developed a rapid diagnostic test for antibiotic-resistant bacteria.", "label": 0},
    {"text": "Oil prices dropped 4% after OPEC+ agreed to gradually increase production quotas.", "label": 0},
    {"text": "A magnitude 6.1 earthquake struck off the coast of New Zealand with no immediate tsunami warning.", "label": 0},
    {"text": "The UK parliament voted to extend renewable energy subsidies through 2035 in a close 312-298 vote.", "label": 0},
    {"text": "Inflation in the eurozone slowed to 2.4% in October, according to Eurostat data released Thursday.", "label": 0},
    {"text": "SpaceX successfully launched 22 Starlink satellites aboard a Falcon 9 rocket from Cape Canaveral.", "label": 0},
]

# Additional hand-crafted fake examples
extra_fake = [
    {"text": "SHOCKING: Scientists discover the moon is actually a hollow space station — NASA has been lying for decades!!!", "label": 1},
    {"text": "URGENT ALERT: New 5G towers are emitting frequencies that ERASE short-term memory — protect your family NOW", "label": 1},
    {"text": "The government is adding LITHIUM to drinking water to make the population EASIER to control — share this truth!", "label": 1},
    {"text": "PROOF: The 2024 election was stolen using quantum computers — whistleblower shares UNDENIABLE evidence", "label": 1},
    {"text": "Big Pharma FURIOUS as man cures terminal cancer with turmeric and coconut oil — doctors don't want you to know", "label": 1},
    {"text": "BREAKING: World leaders met in SECRET to plan population reduction to 500 million — leaked documents reveal all", "label": 1},
    {"text": "NASA scientist FIRED for revealing alien structures on Mars — photos DELETED from official database", "label": 1},
    {"text": "WARNING: This common vegetable DESTROYS your liver — study BURIED by food industry lobbyists", "label": 1},
    {"text": "EXPOSED: Chemtrails contain DNA-altering nanoparticles — Harvard professor SILENCED after publishing findings", "label": 1},
    {"text": "The Federal Reserve is secretly PRINTING trillions to fund a shadow government — insider CONFIRMS", "label": 1},
]

random.seed(42)
real_data = generate_samples(real_templates, real_slots, 800, label=0) + extra_real
fake_data = generate_samples(fake_templates, fake_slots, 800, label=1) + extra_fake

df = pd.DataFrame(real_data + fake_data).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('/home/claude/fakenews_detector/data/news_dataset.csv', index=False)
print(f"Dataset created: {len(df)} samples")
print(df['label'].value_counts())
print(df.head(3))
