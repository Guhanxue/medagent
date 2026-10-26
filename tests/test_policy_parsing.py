"""Action-boundary and bounded-correction regressions; no models or API calls."""
import json
import unittest
from unittest.mock import Mock, patch

from harness.llm_policy import make_llm_policy, parse_action


class ActionParsingTests(unittest.TestCase):
    valid = {"ship", "keep_source", "tent", "escalate"}

    def test_last_complete_nested_action_wins(self):
        first = {"tool": "tent", "args": {"config": {"steps": 3}}}
        last = {"tool": "ship", "args": {"report": {"summary": "done"}}}
        text = json.dumps(first) + "\nRevised decision:\n```json\n" + json.dumps(last) + "\n```"
        self.assertEqual(parse_action(text, self.valid), last)

    def test_braces_and_escaped_quotes_in_report_are_data(self):
        action = {"tool": "ship", "args": {"report": {"summary": 'Keep {x}; quote "tool": "tent" and unmatched }'}}}
        self.assertEqual(parse_action(json.dumps(action), self.valid), action)

    def test_nested_actions_inside_non_action_containers_are_ignored(self):
        nested = {"tool": "tent", "args": {}}
        for container in ({"example": nested}, [nested], {"tool": "unknown", "args": nested}):
            with self.subTest(container=container):
                self.assertIsNone(parse_action(json.dumps(container), self.valid))
                earlier = {"tool": "keep_source", "args": {}}
                self.assertEqual(parse_action(json.dumps(earlier) + json.dumps(container), self.valid), earlier)

    def test_incomplete_or_malformed_outer_containers_hide_nested_actions(self):
        for text in ('{"example": {"tool":"ship","args":{}}',
                     '[{"tool":"ship","args":{}}',
                     '{broken: {"tool":"ship","args":{}}}',
                     '{"example": [{"tool":"ship","args":{}}}'):
            with self.subTest(text=text):
                self.assertIsNone(parse_action(text, self.valid))

    def test_quoted_action_string_is_not_an_action(self):
        self.assertIsNone(parse_action(json.dumps('{"tool":"tent","args":{}}'), self.valid))

    def test_non_dictionary_args_are_rejected(self):
        for args in (None, [], "x", 1, False):
            with self.subTest(args=args):
                self.assertIsNone(parse_action(json.dumps({"tool": "tent", "args": args}), self.valid))

    def test_invalid_tool_types_and_unknown_tools_are_rejected(self):
        for tool in (None, [], {}, 1, False, "unknown"):
            with self.subTest(tool=tool):
                self.assertIsNone(parse_action(json.dumps({"tool": tool, "args": {}}), self.valid))

    def test_missing_args_defaults_to_empty_object(self):
        self.assertEqual(parse_action('{"tool":"tent"}', self.valid), {"tool": "tent", "args": {}})

    def test_invalid_trailing_actions_do_not_replace_last_valid_action(self):
        action = {"tool": "tent", "args": {}}
        text = json.dumps(action) + '{"tool":"unknown","args":{}} {"tool":"ship","args":null}'
        self.assertEqual(parse_action(text, self.valid), action)

    def test_malformed_prefix_and_incomplete_suffix(self):
        self.assertEqual(parse_action('prose {broken}\n{"tool":"tent"}\n{"tool":', self.valid),
                         {"tool": "tent", "args": {}})

    def test_non_text_and_no_action_responses(self):
        for text in (None, [], {}, 7, "", "no action", '[1, 2]', '{"note":"done"}'):
            with self.subTest(text=text):
                self.assertIsNone(parse_action(text, self.valid))


class PolicyCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.view = {"tools": [{"name": "tent"}, {"name": "escalate"}], "evidence": {}}
        prompt = patch("harness.llm_policy.build_prompt", return_value="original task context")
        prompt.start()
        self.addCleanup(prompt.stop)
        fallback = patch("harness.llm_policy._fallback_action", side_effect=AssertionError("fallback forbidden"))
        fallback.start()
        self.addCleanup(fallback.stop)

    def test_valid_response_uses_one_call(self):
        llm = Mock(return_value='{"tool":"tent","args":{}}')
        self.assertEqual(make_llm_policy(llm)(self.view)["tool"], "tent")
        self.assertEqual(llm.call_count, 1)

    def test_invalid_response_gets_one_explicit_correction(self):
        llm = Mock(side_effect=['{"tool":"unknown"}', '{"tool":"tent","args":{"steps":2}}'])
        self.assertEqual(make_llm_policy(llm)(self.view), {"tool": "tent", "args": {"steps": 2}})
        self.assertEqual(llm.call_count, 2)
        correction = llm.call_args_list[1].args[0]
        self.assertIn("original task context", correction)
        self.assertIn("ACTION FORMAT CORRECTION", correction)
        self.assertIn('["escalate", "tent"]', correction)

    def test_two_invalid_responses_escalate_without_tool_substitution(self):
        llm = Mock(return_value='{"tool":"tent","args":[]}')
        action = make_llm_policy(llm)(self.view)
        self.assertEqual(action["tool"], "escalate")
        self.assertEqual(llm.call_count, 2)
        self.assertIn("no fallback action executed", action["args"]["report"]["summary"])

    def test_provider_error_fails_closed_without_retry_or_error_disclosure(self):
        llm = Mock(side_effect=RuntimeError("sensitive provider detail"))
        action = make_llm_policy(llm)(self.view)
        self.assertEqual(action["tool"], "escalate")
        self.assertEqual(llm.call_count, 1)
        self.assertNotIn("sensitive", str(action))

    def test_provider_error_during_correction_fails_closed(self):
        llm = Mock(side_effect=["bad response", RuntimeError("sensitive detail")])
        action = make_llm_policy(llm)(self.view)
        self.assertEqual(action["tool"], "escalate")
        self.assertEqual(llm.call_count, 2)
        self.assertNotIn("sensitive", str(action))

    def test_correction_cannot_select_filtered_tool(self):
        self.view["tools"].append({"name": "load_and_deploy"})
        self.view["deployed"] = True
        llm = Mock(side_effect=["bad response", '{"tool":"load_and_deploy"}'])
        self.assertEqual(make_llm_policy(llm)(self.view)["tool"], "escalate")
        self.assertEqual(llm.call_count, 2)


if __name__ == "__main__":
    unittest.main()
