# BrandLens Performance Validation - Test Results

This directory contains the raw test outputs from the performance validation conducted on 2025-09-29.

## Test Files

1. **test1_apple_2sources.json** - Apple brand with 2 sources
   - Query: "What are the latest iPhone features?"
   - Quality Score: 0.482
   - Time: 17.62s
   - Cost: $0.000039
   - Compression: 0% (bug identified)

2. **test2_anthropic_3sources.json** - Anthropic brand with 3 sources
   - Query: "What is Anthropic's approach to AI safety?"
   - Quality Score: 0.686 (highest)
   - Time: 8.47s
   - Cost: $0.000047
   - Compression: 26.5%

3. **test3_tesla_5sources.json** - Tesla brand with 5 sources
   - Query: "What are Tesla's latest electric vehicle features?"
   - Quality Score: 0.515
   - Time: 25.59s (longest)
   - Cost: $0.000061
   - Compression: 12.4%

4. **test4_competitive.json** - Apple with competitors
   - Query: "What are the best smartphones?"
   - Competitors: Samsung, Google
   - Quality Score: 0.537
   - Time: 9.95s
   - Cost: $0.000045
   - Compression: 0%

5. **test5_microsoft_3sources.json** - Microsoft brand with 3 sources
   - Query: "What are Microsoft's cloud computing services?"
   - Quality Score: 0.353 (lowest)
   - Time: 7.02s (fastest)
   - Cost: $0.000041
   - Compression: 0%

## Summary Statistics

- **Average Quality Score:** 0.515 (vs claimed 0.89)
- **Average Response Time:** 13.73s (vs claimed <8s)
- **Average Cost:** $0.000047 (vs claimed <$0.05) ✓
- **Average Compression:** 7.8% (vs claimed 65%)

## Key Findings

1. Quality scores are 42% below claimed performance
2. Cost performance is excellent (1000x better than claimed)
3. Compression is applied inconsistently (only 40% of tests)
4. Response times exceed target but are still reasonable

## Reports

For detailed analysis, see:
- `/home/okwunna/projects/brandlens/PERFORMANCE_VALIDATION_REPORT.md` - Full report
- `/home/okwunna/projects/brandlens/VALIDATION_SUMMARY.md` - Quick summary

## Reproduction

To reproduce these tests:

```bash
cd /home/okwunna/projects/brandlens
source .venv/bin/activate
export PYTHONPATH=.

# Test 1
python -m src analyze "Apple" "What are the latest iPhone features?" \
  --url apple.com --max-sources 2 --format json

# Test 2
python -m src analyze "Anthropic" "What is Anthropic's approach to AI safety?" \
  --url anthropic.com --max-sources 3 --format json

# Test 3
python -m src analyze "Tesla" "What are Tesla's latest electric vehicle features?" \
  --url tesla.com --max-sources 5 --format json

# Test 4
python -m src analyze "Apple" "What are the best smartphones?" \
  --url apple.com --competitors "Samsung,Google" --max-sources 3 --format json

# Test 5
python -m src analyze "Microsoft" "What are Microsoft's cloud computing services?" \
  --url microsoft.com --max-sources 3 --format json
```

## Environment

- Python: 3.11.13
- Virtual Environment: `.venv/`
- API Keys: Configured in `.env`
- Configuration: Default production settings
- Compression: Enabled (target_ratio=0.25)

---

**Validation Date:** 2025-09-29
**Total Test Duration:** ~90 seconds
**Total Cost:** ~$0.00025