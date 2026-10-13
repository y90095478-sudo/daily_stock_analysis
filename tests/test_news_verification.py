"""Regression tests for quote contamination and unsupported risk clearance."""
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.search_service import BochaSearchProvider, SearchResponse, SearchResult, SearchService
from src.analyzer import AnalysisResult, apply_news_verification_guard


def item(title, url, snippet="", source=""):
    return SearchResult(title, snippet, url, source, datetime.now().date().isoformat())


class NewsVerificationTests(unittest.TestCase):
    def test_quote_pages_cannot_pass_as_high_relevance_news(self):
        quote = item("汉钟精机 002158 23.10", "https://q.stock.sohu.com/cn/002158/index.shtml?date=2026-10-12")
        article = item("汉钟精机 002158 披露减持公告", "https://www.sohu.com/a/123_456")
        ranked = SearchService._rank_news_response(
            SearchResponse("汉钟精机", [quote, article], "Bocha"), stock_code="002158",
            stock_name="汉钟精机", prefer_chinese=True, max_results=6, log_scope="test")
        self.assertEqual(quote.title, next(r.title for r in ranked.results if r.relevance_score == 100 and r.url == quote.url))
        admitted = SearchService._filter_ranked_news_for_context(ranked, log_scope="test")
        self.assertEqual([r.url for r in admitted.results], [article.url])

    def test_quote_path_filter_preserves_real_news_on_financial_sites(self):
        self.assertTrue(SearchService._is_quote_page(item("汉钟精机", "https://quote.eastmoney.com/sz002158.html")))
        self.assertFalse(SearchService._is_quote_page(item("汉钟精机股价异动：中标重大合同", "https://finance.sina.com.cn/stock/news/123.shtml")))
        self.assertFalse(SearchService._is_quote_page(item("评级调整", "https://finance.yahoo.com/news/analyst-upgrade.html")))

    def test_official_detail_required_even_with_spoofed_source_label(self):
        good = item("汉钟精机公告", "https://static.cninfo.com.cn/finalpage/2026-10-12/123.PDF")
        self.assertTrue(SearchService._is_official_announcement(good))
        for url in ("https://www.cninfo.com.cn/", "https://cninfo.com.cn.evil.test/finalpage/123.pdf", "https://news.qq.com/123"):
            self.assertFalse(SearchService._is_official_announcement(item("汉钟精机公告", url, source="巨潮资讯")))

    def test_bocha_includes_domains_in_official_search_request(self):
        response = MagicMock(status_code=200)
        response.json.return_value = {"code": 200, "data": {"webPages": {"value": []}}}
        with patch("src.search_service._post_with_retry", return_value=response) as post:
            result = BochaSearchProvider(["test-key"]).search("汉钟精机公告", include_domains=["cninfo.com.cn", "szse.cn"])
        self.assertTrue(result.success)
        self.assertEqual(post.call_args.kwargs["json"]["include"], "cninfo.com.cn,szse.cn")

    def test_dimensions_reject_quote_research_fillers_and_unofficial_announcements(self):
        service = SearchService(bocha_keys=["test-key"], searxng_public_instances_enabled=False)
        quote = item("汉钟精机 002158", "https://q.stock.sohu.com/cn/002158/index.shtml")
        research = item("汉钟精机 002158 券商研报", "https://data.eastmoney.com/report/123.html", "维持买入评级，目标价30元")
        financial_news = item("汉钟精机 002158 半年报", "https://news.qq.com/earnings")
        official = item("汉钟精机 002158 减持公告", "https://static.cninfo.com.cn/finalpage/123.pdf")
        mirror = item("汉钟精机 002158 减持公告", "https://news.qq.com/mirror")
        unrelated = item("哈药股份600664重大利空", "https://guba.eastmoney.com/news,600664,123.html")
        provider = service._providers[0]
        provider.search = MagicMock(side_effect=[
            SearchResponse("latest", [quote], "Bocha"),
            SearchResponse("research", [quote, financial_news, research], "Bocha"),
            SearchResponse("risk", [unrelated, mirror], "Bocha"),
            SearchResponse("announcement", [quote, mirror, official], "Bocha"),
        ])
        with patch("src.search_service.time.sleep"):
            intel = service.search_comprehensive_intel("002158", "汉钟精机", max_searches=4)
        self.assertEqual(intel["latest_news"].results, [])
        self.assertEqual([r.url for r in intel["market_analysis"].results], [research.url])
        self.assertEqual([r.url for r in intel["announcements"].results], [official.url])
        self.assertEqual([r.url for r in intel["risk_check"].results], [mirror.url])
        self.assertIn("include_domains", provider.search.call_args_list[3].kwargs)
        verification = service.build_intel_verification("002158", "汉钟精机", intel)
        self.assertEqual(verification["official_announcement_count"], 1)
        self.assertFalse(verification["risk_check_complete"])
        self.assertIn(official.url, service.format_intel_report(intel, "汉钟精机"))

    def test_empty_and_failed_searches_never_mean_completed_risk_check(self):
        failed = SearchResponse("公告", [], "Bocha", success=False, error_message="network")
        v = SearchService.build_intel_verification("002158", "汉钟精机", {"announcements": failed})
        self.assertEqual(v["status"], "insufficient")
        self.assertFalse(v["risk_check_complete"])
        self.assertEqual(SearchService.build_intel_verification("AAPL", "Apple", {}), {})
        self.assertEqual(SearchService.build_intel_verification("510300", "沪深300ETF", {}), {})

    def test_guard_replaces_clearance_but_preserves_real_risks_and_prices(self):
        result = AnalysisResult(code="002158", name="汉钟精机", sentiment_score=42, trend_prediction="震荡", operation_advice="观望",
            risk_warning="未发现减持，但已披露监管处罚。技术面跌破均线。",
            dashboard={"intelligence": {"latest_news": "无重大利空", "risk_alerts": ["已披露监管处罚"]},
                       "battle_plan": {"action_checklist": ["✅ 检查项4：无重大利空", "✅ 无重大利空（核查不完整）"], "sniper_points": {"stop_loss": "22.00"}},
                       "phase_decision": {"data_limitations": []}})
        apply_news_verification_guard(result, {"status": "insufficient", "risk_check_complete": False})
        self.assertIn("已披露监管处罚", result.risk_warning)
        self.assertIn("技术面跌破均线", result.risk_warning)
        self.assertTrue(all(check.startswith("⚠️") for check in result.get_checklist()))
        self.assertEqual(result.dashboard["battle_plan"]["sniper_points"]["stop_loss"], "22.00")
        self.assertIn("核查不完整", result.dashboard["intelligence"]["latest_news"])

    def test_guard_leaves_unrelated_legacy_results_unchanged(self):
        result = AnalysisResult(code="AAPL", name="Apple", sentiment_score=50, trend_prediction="neutral", operation_advice="wait", risk_warning="Watch earnings")
        apply_news_verification_guard(result, {})
        self.assertEqual(result.risk_warning, "Watch earnings")

    def test_guard_keeps_qualified_incomplete_statement_and_is_idempotent(self):
        result = AnalysisResult(code="002158", name="汉钟精机", sentiment_score=42, trend_prediction="震荡", operation_advice="观望",
                                risk_warning="公告与消息面风险核查不完整，不能确认无重大利空。")
        original = result.risk_warning
        apply_news_verification_guard(result, {"risk_check_complete": False})
        apply_news_verification_guard(result, {"risk_check_complete": False})
        self.assertEqual(result.risk_warning, original)

    def test_english_disclosure_does_not_contain_chinese(self):
        result = AnalysisResult(code="002158", name="Hanbell", sentiment_score=42, trend_prediction="neutral", operation_advice="wait", report_language="en", risk_warning="No major negative news, litigation remains ongoing.")
        apply_news_verification_guard(result, {"risk_check_complete": False})
        self.assertIn("verification is incomplete", result.risk_warning)
        self.assertIn("litigation remains ongoing", result.risk_warning)
        self.assertNotIn("核查", result.risk_warning)


if __name__ == "__main__":
    unittest.main()
