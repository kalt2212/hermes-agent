"""An approval card whose command preview is cut never offers an approve choice for text the user could
not read: the whole command is sent just before the card, or, when it cannot be, only Deny is offered."""

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult


class _Chat(BasePlatformAdapter):
    """Concrete adapter with a 1,000-char cap that records sends and the rendered prompt."""

    MAX_MESSAGE_LENGTH = 1000

    def __init__(self, *, send_ok: bool = True) -> None:
        self.sent, self.prompts, self._send_ok = [], [], send_ok

    def _exec_approval_cmd_budget(self, description: str, smart_denied: bool) -> int:
        return 600

    async def connect(self):  # pragma: no cover - unused
        pass

    async def disconnect(self):  # pragma: no cover - unused
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        self.sent_metadata = metadata
        return SendResult(success=self._send_ok, message_id="m1")

    async def _send_exec_approval_prompt(self, prompt):
        self.prompts.append(prompt)
        return SendResult(success=True, message_id="card")


def _choices(adapter):
    return [choice for _, choice, _ in adapter.prompts[-1].actions]


async def _ask(adapter, command):
    await adapter.send_exec_approval("c1", command, "sk", description="flagged", request_id="r1")
    return adapter.prompts[-1].text


@pytest.mark.asyncio
async def test_short_command_keeps_the_compact_card():
    adapter = _Chat()
    text = await _ask(adapter, "rm -rf /tmp/x")
    assert adapter.sent == [] and "rm -rf /tmp/x" in text and "Preview shortened" not in text
    assert _choices(adapter) == ["once", "session", "always", "deny"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "send_ok", "choices", "attempted"), [
    ("echo " + "A" * 900 + " && rm -rf ~ # TAIL", True, ["once", "session", "always", "deny"], True),
    ("echo " + "A" * 900 + " && rm -rf ~ # TAIL", False, ["deny"], True),  # the full send failed
    ("echo '```' " + "A" * 900 + " && rm -rf ~ # TAIL", True, ["deny"], False),  # cannot be fenced verbatim
])
async def test_truncated_preview_is_backed_by_the_full_command_or_deny_only(command, send_ok, choices, attempted):
    adapter = _Chat(send_ok=send_ok)
    text = await _ask(adapter, command)
    assert "TAIL" not in text and "Preview shortened" in text and str(len(command)) in text
    assert _choices(adapter) == choices
    assert adapter.sent == ([f"```\n{command}\n```"] if attempted else [])


@pytest.mark.asyncio
async def test_full_command_send_is_marked_interim_so_it_never_seals_the_live_response():
    """The full command goes out mid-turn: unmarked, a stream-is-the-message adapter (relay drafts) reads it
    as the turn-final reply and swallows every later update to the open stream."""
    adapter = _Chat()
    await adapter.send_exec_approval("c1", "echo " + "A" * 900, "sk", metadata={"thread_id": "t1"})
    assert adapter.sent_metadata == {"thread_id": "t1", "_interim_send": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_factory", ["base", "discord", "slack"])
async def test_backticks_inside_a_short_command_cannot_close_the_card_fence(adapter_factory):
    """A ``` (or longer run) in the command would end the card's markdown fence early, and the rest
    would render as markdown (spoilers, links): the fence is longer than any run in the command."""
    from gateway.platforms.base_exec_approval import render_exec_approval

    adapter = _Chat()
    if adapter_factory == "discord":
        from plugins.platforms.discord.adapter import DiscordAdapter
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        adapter._approval_mention_content = lambda: ""
    elif adapter_factory == "slack":
        from plugins.platforms.slack.adapter import SlackAdapter
        adapter = SlackAdapter.__new__(SlackAdapter)
    command = "echo '```' ||spoiler|| [x](https://e.x) ````` done"
    text, truncated = render_exec_approval(adapter, command, "flagged", False, True)
    assert not truncated
    fence = "``````"  # one longer than the command's longest run (5)
    assert text.count(fence) == 2
    assert text.index(fence) < text.index(command) < text.rindex(fence)


@pytest.mark.parametrize("command", ["echo " + "`" * 900 + " ; rm -rf ~", "`" * 1990, "`" * 5000],
                         ids=["echo-900-ticks", "1990-ticks", "5000-ticks"])
def test_a_long_backtick_run_the_preview_drops_cannot_push_the_card_past_the_cap(command):
    """The fence is sized from the preview actually shown: a backtick run cut off with the tail must not
    widen the fence past Discord's 2,000-char cap (that card fell back to text and lost its buttons)."""
    import re

    from gateway.platforms.base_exec_approval import render_exec_approval
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._approval_mention_content = lambda: ""
    text, truncated = render_exec_approval(adapter, command, "flagged", False, False)
    assert len(text) <= adapter.MAX_MESSAGE_LENGTH
    assert truncated and "Preview shortened" in text and "only Deny is offered" in text
    runs = re.findall(r"^(`{3,})", text, re.MULTILINE)
    open_fence, close_fence = runs[0], runs[-1]  # the opener line and the closer line
    shown = text[text.index(open_fence) + len(open_fence):text.rindex(close_fence)].removeprefix("bash")
    assert open_fence == close_fence and max(map(len, re.findall("`+", shown)), default=0) < len(open_fence)


@pytest.mark.parametrize("command", ["`" * 20 + " rm -rf ~ " + "`" * 20, "`" * 20 + "x" * 4000 + "`" * 20],
                         ids=["fits", "cut"])
def test_slack_fence_sits_on_its_own_line_so_the_commands_backticks_never_merge_into_it(command):
    """With the fence glued to the command, 20 leading backticks and a 21-tick fence render as one 41-tick run."""
    import re

    from gateway.platforms.base_exec_approval import render_exec_approval
    from plugins.platforms.slack.adapter import SlackAdapter

    adapter = SlackAdapter.__new__(SlackAdapter)
    text, _ = render_exec_approval(adapter, command, "flagged", False, False)
    assert len(text) <= 3000  # Slack's section cap
    fence = "`" * 21
    assert text.splitlines().count(fence) == 2  # opener and closer each alone on a line
    shown = text.split(fence + "\n", 1)[1].rsplit("\n" + fence, 1)[0]
    assert shown.startswith("`" * 20) and max(map(len, re.findall("`+", shown))) == 20


# ---- quoted heredoc collapse (#158342) ---------------------------------------------------------------------------


def _heredoc_command(opener="cat > /tmp/body.md <<'EOF'", body_lines=300, terminator="EOF"):
    body = "\n".join(f"line {i} of the issue body with some words" for i in range(body_lines))
    return f"{opener}\n{body}\n{terminator}\ngh issue create -R owner/repo --body-file /tmp/body.md"


def test_collapse_keeps_opener_terminator_and_tail_for_a_quoted_heredoc():
    from gateway.platforms.base_exec_approval import collapse_quoted_heredocs

    command = _heredoc_command()
    assert collapse_quoted_heredocs(command) == (
        "cat > /tmp/body.md <<'EOF'\n[heredoc: 300 lines collapsed]\nEOF\n"
        "gh issue create -R owner/repo --body-file /tmp/body.md")


@pytest.mark.parametrize("opener", ["cat > /tmp/b.md <<\"EOF\"", "tee /tmp/b.md <<'EOF' >/dev/null"])
def test_collapse_accepts_double_quoted_delimiter_and_tee(opener):
    from gateway.platforms.base_exec_approval import collapse_quoted_heredocs

    assert "[heredoc: 300 lines collapsed]" in collapse_quoted_heredocs(_heredoc_command(opener=opener))


@pytest.mark.parametrize("command", [
    _heredoc_command(opener="cat > /tmp/b.md <<EOF"),  # unquoted: the shell expands the body
    _heredoc_command(opener="bash <<'EOF'"),  # quoted, but the body is run as code, not data
    _heredoc_command(opener="cat <<'EOF' | sh"),  # piped to a shell
    _heredoc_command(opener="python3 <<'EOF'"),  # piped to an interpreter
    "cat <<'EOF'\nrun $(whoami)\nEOF\ngh issue create",  # command substitution in the body
    "cat <<'EOF'\nuse `date` here\nEOF\ngh issue create",  # backtick in the body
    "cat <<'EOF'\nno terminator here\ngh issue create",  # unterminated: nothing to delimit
])
def test_collapse_leaves_unsafe_or_unterminated_heredocs_untouched(command):
    from gateway.platforms.base_exec_approval import collapse_quoted_heredocs

    assert collapse_quoted_heredocs(command) == command


def test_collapse_handles_dash_heredoc_with_tab_indented_terminator():
    from gateway.platforms.base_exec_approval import collapse_quoted_heredocs

    command = "cat <<-'EOF'\n\tbody one\n\tbody two\n\tEOF\ngh issue create"
    assert collapse_quoted_heredocs(command) == "cat <<-'EOF'\n[heredoc: 2 lines collapsed]\n\tEOF\ngh issue create"


def test_collapse_is_a_no_op_for_commands_without_heredocs():
    from gateway.platforms.base_exec_approval import collapse_quoted_heredocs

    command = "echo hi && rm -rf /tmp/x"
    assert collapse_quoted_heredocs(command) == command


@pytest.mark.asyncio
async def test_long_quoted_heredoc_card_shows_the_command_tail():
    """The reported case: a 2 KB heredoc body used to push the gh call past the cut. Now the body is
    collapsed, the tail is on the card, and the card still counts as cut (full command or Deny only)."""
    adapter = _Chat()
    command = _heredoc_command()
    text = await _ask(adapter, command)
    assert "[heredoc: 300 lines collapsed]" in text
    assert "gh issue create -R owner/repo --body-file /tmp/body.md" in text
    assert "line 150 of the issue body" not in text
    assert adapter.sent == [f"```\n{command}\n```"]  # the whole command still goes out first
    assert _choices(adapter) == ["once", "session", "always", "deny"]
