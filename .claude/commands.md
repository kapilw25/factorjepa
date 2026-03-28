# Claude Code — Built-in Commands

> Complete reference for built-in commands available in Claude Code.
> Documentation index: https://code.claude.com/docs/llms.txt

Type `/` to see all commands, or `/` + letters to filter.
`<arg>` = required, `[arg]` = optional.

---

## Session & Navigation

| Command | Purpose |
|---------|---------|
| `/clear` | Clear conversation history. Aliases: `/reset`, `/new` |
| `/exit` | Exit the CLI. Alias: `/quit` |
| `/resume [session]` | Resume conversation by ID/name. Alias: `/continue` |
| `/branch [name]` | Branch conversation at this point. Alias: `/fork` |
| `/rewind` | Rewind conversation/code to previous point. Alias: `/checkpoint` |
| `/rename [name]` | Rename current session |
| `/add-dir <path>` | Add working directory to session |
| `/compact [instructions]` | Compact conversation with optional focus |
| `/export [filename]` | Export conversation as plain text |
| `/copy [N]` | Copy last response to clipboard |

## Model & Configuration

| Command | Purpose |
|---------|---------|
| `/model [model]` | Select or change AI model |
| `/effort [low\|medium\|high\|max\|auto]` | Set model effort level |
| `/fast [on\|off]` | Toggle fast mode |
| `/config` | Open Settings (theme, model, output style). Alias: `/settings` |
| `/color [color\|default]` | Set prompt bar color (`red`, `blue`, `green`, etc.) |
| `/theme` | Change color theme (light/dark/colorblind) |
| `/vim` | Toggle Vim/Normal editing modes |
| `/permissions` | View/update permissions. Alias: `/allowed-tools` |
| `/sandbox` | Toggle sandbox mode |

## Context & Memory

| Command | Purpose |
|---------|---------|
| `/context` | Visualize context usage as colored grid |
| `/memory` | Edit CLAUDE.md, enable/disable auto-memory |
| `/btw <question>` | Ask side question without adding to conversation |
| `/plan [description]` | Enter plan mode (e.g., `/plan fix the auth bug`) |

## Code & Git

| Command | Purpose |
|---------|---------|
| `/diff` | Interactive diff viewer (uncommitted changes + per-turn diffs) |
| `/pr-comments [PR]` | Fetch GitHub PR comments. Requires `gh` CLI |
| `/security-review` | Analyze branch changes for security vulnerabilities |
| `/review` | Deprecated. Use `code-review` plugin instead |

## Integrations

| Command | Purpose |
|---------|---------|
| `/ide` | Manage IDE integrations |
| `/desktop` | Continue session in Desktop app. Alias: `/app` |
| `/chrome` | Configure Claude in Chrome |
| `/install-github-app` | Set up Claude GitHub Actions |
| `/install-slack-app` | Install Claude Slack app |
| `/mcp` | Manage MCP server connections |
| `/remote-control` | Enable remote control from claude.ai. Alias: `/rc` |
| `/remote-env` | Configure remote environment for web sessions |

## Extensions & Skills

| Command | Purpose |
|---------|---------|
| `/plugin` | Manage Claude Code plugins |
| `/reload-plugins` | Reload all active plugins |
| `/skills` | List available skills |
| `/agents` | Manage agent configurations |
| `/hooks` | View hook configurations |
| `/init` | Initialize project with CLAUDE.md guide |
| `/keybindings` | Open keybindings config file |
| `/terminal-setup` | Configure terminal keybindings (Shift+Enter, etc.) |
| `/statusline` | Configure status line |
| `/schedule [description]` | Create/manage Cloud scheduled tasks |

## Account & Usage

| Command | Purpose |
|---------|---------|
| `/login` | Sign in to Anthropic account |
| `/logout` | Sign out |
| `/usage` | Show plan usage limits and rate limit status |
| `/cost` | Show token usage statistics |
| `/extra-usage` | Configure extra usage for rate limits |
| `/upgrade` | Open upgrade page |
| `/privacy-settings` | View/update privacy settings (Pro/Max only) |
| `/passes` | Share free week of Claude Code |
| `/voice` | Toggle push-to-talk voice dictation |

## Info & Diagnostics

| Command | Purpose |
|---------|---------|
| `/help` | Show help and available commands |
| `/status` | Show version, model, account, connectivity |
| `/doctor` | Diagnose and verify installation |
| `/stats` | Visualize daily usage, session history, streaks |
| `/insights` | Generate report analyzing your sessions |
| `/release-notes` | View full changelog |
| `/feedback [report]` | Submit feedback. Alias: `/bug` |
| `/mobile` | QR code for mobile app. Aliases: `/ios`, `/android` |
| `/stickers` | Order Claude Code stickers |

---

## MCP Prompts

MCP servers expose prompts as commands using the format `/mcp__<server>__<prompt>`, dynamically discovered from connected servers.

## See Also

- [Skills](https://code.claude.com/en/skills): create your own commands
- [Interactive mode](https://code.claude.com/en/interactive-mode): keyboard shortcuts, Vim mode, command history
- [CLI reference](https://code.claude.com/en/cli-reference): launch-time flags
