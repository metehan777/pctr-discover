const QUALITY_WEIGHTS = {
  entity_density:      0.22,
  topic_clarity:       0.18,
  informational_value: 0.16,
  freshness_signal:    0.12,
  engagement_depth:    0.10,
  title_formatting:    0.08,
  natural_authority:   0.08,
  visual_promise:      0.06,
};

const BETA_WEIGHT    = 0.35;
const CTR_FLOOR      = 0.005;
const CTR_CEIL       = 0.22;
const SIGMOID_ALPHA  = 0.65;
const SIGMOID_MU     = 5.5;
const MODEL          = "claude-opus-4-6";
const FALLBACK_MODEL = "gpt-5-nano";

function computePctr(scores, clickbaitScore) {
  let quality = 0;
  for (const k in QUALITY_WEIGHTS) {
    quality += QUALITY_WEIGHTS[k] * scores[k];
  }

  const betaPenalty = 1.0 - (BETA_WEIGHT * clickbaitScore / 10.0);
  const raw = quality * betaPenalty;
  const z = SIGMOID_ALPHA * (raw - SIGMOID_MU);
  const sigmoid = 1.0 / (1.0 + Math.exp(-z));
  const pctr = CTR_FLOOR + (CTR_CEIL - CTR_FLOOR) * sigmoid;

  const contributions = {};
  for (const k in QUALITY_WEIGHTS) {
    contributions[k] = Math.round(QUALITY_WEIGHTS[k] * scores[k] * 1000) / 1000;
  }

  return {
    pctr:          Math.round(pctr * 100000) / 100000,
    pctr_pct:      Math.round(pctr * 10000) / 100,
    quality_score: Math.round(quality * 100) / 100,
    beta_penalty:  Math.round(betaPenalty * 1000) / 1000,
    raw_score:     Math.round(raw * 100) / 100,
    sigmoid_input: Math.round(z * 1000) / 1000,
    contributions,
  };
}

const ANALYSIS_PROMPT = `You are a Google Discover pCTR analyst. You understand Discover's 9-stage content pipeline and how its pCTR model evaluates titles.

KEY CONTEXT about Google Discover's actual ranking:
- Discover's pCTR model uses og:title as a DIRECT input
- Content is matched to users via Knowledge Graph entity MIDs (Stage 5)
- The feedback loop (Stage 9) tracks engagement_time_msec and PENALIZES clickbait
- Titles that get clicks but quick bounces see REDUCED future pCTR
- Google calls this the "blindness" factor — users learn to skip low-quality patterns
- Entity-rich, informative titles that lead to genuine engagement score HIGHEST over time

Analyze the following news title. Score each dimension 0.0 to 10.0. Be calibrated:
- Titles with strong named entities and clear topics should score HIGH
- Clickbait patterns (withholding info, exaggeration, misleading) should score HIGH on clickbait_score
- A good Discover title is informative, entity-rich, and naturally written — NOT clickbaity

DIMENSIONS (positive — higher is better):

1. **entity_density** (weight: 22%)
   How many recognizable Knowledge Graph entities does the title contain?
   (People, brands, organizations, places, products, technologies, events)
   0 = no identifiable entities, 10 = multiple strong KG entities

2. **topic_clarity** (weight: 18%)
   How clearly does the title signal its topic for Discover's content classification?
   Can the system immediately classify this into a Discover cluster type?
   0 = ambiguous/unclear topic, 10 = instantly classifiable

3. **informational_value** (weight: 16%)
   How much substantive information does the title convey?
   Will clicking lead to genuine content engagement (high engagement_time_msec)?
   0 = no real information, 10 = rich, substantive preview

4. **freshness_signal** (weight: 12%)
   Does the title signal breaking/recent news? (Discover's 1-7 day bucket = highest weight)
   0 = no freshness signal, 10 = clearly breaking/just-happened news

5. **engagement_depth** (weight: 10%)
   Predict: will a user who clicks spend significant time reading?
   0 = likely quick bounce, 10 = likely deep engagement

6. **title_formatting** (weight: 8%)
   Is the title length optimal for Discover cards? (60-100 chars ideal)
   Is punctuation/capitalization natural and well-formatted?
   0 = poor formatting, 10 = optimal Discover card display

7. **natural_authority** (weight: 8%)
   Does the title use natural, trustworthy language? Does it carry authority cues?
   (Named sources, institutional references, expert attribution)
   0 = untrustworthy/sensational, 10 = authoritative and natural

8. **visual_promise** (weight: 6%)
   Does the title imply visual/media content that would pair well with a Discover card image?
   0 = purely abstract, 10 = strongly implies visual content

PENALTY DIMENSION (negative — higher means MORE clickbait → LOWER pCTR):

9. **clickbait_score** (β blindness factor)
   How clickbaity is this title? Score the presence of these patterns:
   - Withholding key information to force clicks ("You won't believe...")
   - Exaggeration/sensationalism beyond what facts support
   - Misleading framing (title promises something content can't deliver)
   - Listicle bait ("10 things that...")
   - Emotional manipulation without substance
   0 = zero clickbait patterns (good), 10 = extreme clickbait (bad for Discover)

ALSO PROVIDE:
- Brief 1-2 sentence analysis
- 1-2 Discover-specific improvements

Respond ONLY with valid JSON (no markdown fences):
{
  "scores": {
    "entity_density": <float>,
    "topic_clarity": <float>,
    "informational_value": <float>,
    "freshness_signal": <float>,
    "engagement_depth": <float>,
    "title_formatting": <float>,
    "natural_authority": <float>,
    "visual_promise": <float>
  },
  "clickbait_score": <float>,
  "analysis": "<string>",
  "improvements": ["<string>", "<string>"]
}`;

function parseAnalysisResponse(rawText) {
  let text = rawText.trim();
  if (text.startsWith("```")) {
    text = text.replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");
  }
  return JSON.parse(text);
}

function buildResult(title, parsed) {
  const scores = parsed.scores;
  const clickbait = Math.max(0, Math.min(10, parseFloat(parsed.clickbait_score)));

  for (const key in QUALITY_WEIGHTS) {
    scores[key] = Math.max(0, Math.min(10, parseFloat(scores[key])));
  }

  const pctrResult = computePctr(scores, clickbait);

  return {
    title,
    scores,
    clickbait_score: clickbait,
    analysis: parsed.analysis || "",
    improvements: parsed.improvements || [],
    ...pctrResult,
  };
}

async function callClaude(title, apiKey) {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type":      "application/json",
      "x-api-key":         apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 600,
      temperature: 0,
      messages: [
        { role: "user", content: `${ANALYSIS_PROMPT}\n\nTITLE: "${title}"` },
      ],
    }),
  });

  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`Anthropic API ${resp.status}: ${err}`);
  }

  const msg = await resp.json();
  return msg.content[0].text;
}

async function callOpenAI(title, apiKey) {
  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: FALLBACK_MODEL,
      temperature: 0,
      max_tokens: 600,
      messages: [
        { role: "user", content: `${ANALYSIS_PROMPT}\n\nTITLE: "${title}"` },
      ],
    }),
  });

  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`OpenAI API ${resp.status}: ${err}`);
  }

  const msg = await resp.json();
  return msg.choices[0].message.content;
}

async function analyzeTitle(title, anthropicKey, openaiKey) {
  let rawText;
  try {
    rawText = await callClaude(title, anthropicKey);
  } catch (claudeErr) {
    if (!openaiKey) throw claudeErr;
    try {
      rawText = await callOpenAI(title, openaiKey);
    } catch (oaiErr) {
      throw new Error(`Claude failed: ${claudeErr.message} | OpenAI fallback failed: ${oaiErr.message}`);
    }
  }

  const parsed = parseAnalysisResponse(rawText);
  return buildResult(title, parsed);
}

export async function onRequestPost(context) {
  const anthropicKey = context.env.ANTHROPIC_API_KEY;
  const openaiKey    = context.env.OPENAI_API_KEY;

  if (!anthropicKey && !openaiKey) {
    return Response.json({ error: "No API keys configured" }, { status: 500 });
  }

  let body;
  try {
    body = await context.request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const titles = (body.titles || [])
    .map((t) => (typeof t === "string" ? t.trim() : ""))
    .filter(Boolean)
    .map((t) => t.slice(0, 150))
    .slice(0, 5);

  if (!titles.length) {
    return Response.json({ error: "No titles provided" }, { status: 400 });
  }

  const results = [];
  for (const title of titles) {
    try {
      results.push(await analyzeTitle(title, anthropicKey, openaiKey));
    } catch (e) {
      results.push({ title, error: e.message });
    }
  }

  return Response.json({ results });
}
