# 37signals Projects — Deep Code Analysis

---

## 1. Campfire (Group Chat)

### Domain Model

Single-tenant (one Account enforced by `singleton_guard` unique index). The core schema is tight — 12 tables total (excluding Rails internals):

```
Account (singleton) → Rooms (STI: Open/Closed/Direct) → Messages → Boosts
                    → Users (roles: member/administrator/bot) → Memberships → Push::Subscriptions
                    → Bans, Sessions, Searches, Webhooks
```

**STI for Rooms** — `Rooms::Open`, `Rooms::Closed`, `Rooms::Direct` all live in the `rooms` table with a `type` column. Each subclass has behavior differences:
- `Open` rooms auto-grant membership to all users on creation (`after_save_commit :grant_access_to_all_users`)
- `Direct` rooms are singleton per user-pair — `find_or_create_for` iterates all direct rooms checking user sets (there's even a `FIXME` comment acknowledging this won't scale past 10K+ DMs)
- `Closed` rooms are just `Room` with no extras

**Messages** use Action Text (`has_rich_text :body`) plus a separate `attachment` for file uploads. Each message gets a `client_message_id` (UUID) for optimistic rendering — the client sends a pending message to the DOM before server confirmation arrives. Messages also have `Boosts` (emoji reactions, 16-char limit).

### Authentication

Hand-rolled, no Devise. The pattern:
- `Session` model with `has_secure_token` — stored in a signed permanent cookie (`cookies.signed.permanent[:session_token]`)
- `Authentication` concern on `ApplicationController` — runs `before_action :require_authentication` by default
- Login: `User.authenticate_by(email_address:, password:)` → `start_new_session_for(user)`
- Sessions track `ip_address`, `user_agent`, `last_active_at` (refreshed at most once per hour via `ACTIVITY_REFRESH_RATE`)
- Rate limiting on login: `rate_limit to: 10, within: 3.minutes, only: :create` (Rails 8 built-in)
- CSRF protection disabled for bot API requests: `protect_from_forgery with: :exception, unless: -> { authenticated_by.bot_key? }`

**Bot auth** is separate — token-based (`bot_key = "#{id}-#{bot_token}"`) sent as a URL param, not a cookie.

### Authorization

Minimal. `User::Role` has three enum values: `member`, `administrator`, `bot`. The `can_administer?` method checks admin status OR if you're the creator:
```ruby
def can_administer?(record = nil)
  administrator? || self == record&.creator || record&.new_record?
end
```

**Ban system** is IP-based — when you ban a user, all their session IPs get recorded in the `bans` table, then `BlockBannedRequests` concern rejects any non-GET request from banned IPs. Bans validate the IP is public (no banning localhost/private ranges).

### Real-Time Architecture

Six Action Cable channels, each with a specific purpose:
- **RoomChannel** — subscribes to a room, streams Turbo Stream broadcasts (message append/remove)
- **PresenceChannel** (extends RoomChannel) — tracks user connections. Uses a `Connectable` concern with a connection counter and 60-second TTL. When you subscribe, it calls `membership.present` (increments connection count + touches `connected_at`). Unsubscribe calls `membership.absent`.
- **TypingNotificationsChannel** — broadcasts `{action: :start/stop, user: {id, name}}` to the room
- **HeartbeatChannel** — empty channel (likely just for keep-alive)
- **UnreadRoomsChannel** — global broadcast when any room gets a new message (`ActionCable.server.broadcast("unread_rooms", ...)`)
- **ReadRoomsChannel** — per-user stream for marking rooms as read

**Message flow:** `after_create_commit -> { room.receive(self) }` → Room marks memberships as unread (only disconnected, visible members) → queues `PushMessageJob` for Web Push → Turbo Stream appends the message.

### JavaScript Architecture

Sophisticated Stimulus controllers with private class fields (modern JS). Key patterns:

**`messages_controller.js`** is the heart — 180+ lines managing:
- `MessagePaginator` — infinite scroll pagination, trims excess messages when too many accumulate
- `ScrollManager` — serializes scroll operations through a promise chain (`#pendingOperations`). Auto-scrolls if near bottom, shows "new messages" indicator if not
- `MessageFormatter` — applies CSS classes for threading, @mentions, emoji-only messages
- `ClientMessage` — optimistic rendering from a template, handles pending/failed states
- Turbo Stream interception: `beforeStreamRender` event handler that wraps the render in scroll management

**`client_message.js`** detects emoji-only messages via regex (`/^(\p{Emoji_Presentation}|\p{Extended_Pictographic}|\uFE0F)+$/gu`) and play commands (`/play soundname`). Contains a hardcoded list of 56 sound effects (56k, ballmer, bezos, bueller, clowntown, cottoneyejoe, etc.) — classic Campfire.

### Full-Text Search

SQLite FTS5 virtual table (`message_search_index`) maintained via manual SQL:
```ruby
after_create_commit  :create_in_index
# ...
def create_in_index
  execute_sql_with_binds "insert into message_search_index(rowid, body) values (?, ?)", id, plain_text_body
end
```
Uses porter tokenizer. Searches join messages to the FTS index.

### SSRF Protection

`RestrictedHTTP::PrivateNetworkGuard` — resolves hostnames via `Resolv`, blocks private/loopback/link-local/ipv4-mapped/ipv4-compat IPs. Used for link unfurling (OpenGraph metadata fetch). A recent commit specifically blocked IPv6 `ipv4_compat` addresses as a bypass vector.

### Web Push

Custom implementation in `lib/web_push/` — a push pool that queues payloads for delivery. Push subscriptions are per-device (tracked by endpoint + keys). The `MessagePusher` differentiates push content between DMs (shows sender name as title) and rooms (shows room name as title).

### CSS

26 hand-written CSS files (no framework). Component-based naming: `messages.css`, `composer.css`, `sidebar.css`, `boosts.css`. Custom CSS injection via `Account#custom_styles` (stored as text in the accounts table).

---

## 2. Fizzy (Kanban / Issue Tracker)

### Domain Model

Multi-tenant with full account separation. 69 database tables. Core hierarchy:

```
Identity (cross-account, email-based) → Users (per-account) → Accounts
Account → Boards → Cards → Comments, Reactions, Steps, Taggings
                 → Columns (kanban columns, positioned)
                 → Webhooks → Deliveries
Card → Assignments, Closures, Goldness, NotNow, Pins, Watches, ActivitySpikes
     → Events (polymorphic activity log) → Notifications → NotificationBundles
```

**Identity vs User split** — `Identity` is the cross-account entity (has email, passkeys, avatar, sessions). `User` belongs to both `Identity` and `Account`. This enables one person to be a member of multiple accounts. The `Current` object chains: `session → identity → user (looked up by identity + account)`.

**Cards are the central entity** — 20+ concerns mixed in:
```ruby
include Accessible, Assignable, Attachments, Broadcastable, Closeable, Colored, Commentable,
  Entropic, Eventable, Exportable, Golden, Mentions, Multistep, Pinnable, Postponable, Promptable,
  Readable, Searchable, Stallable, Statuses, Storage::Tracked, Taggable, Triageable, Watchable
```

Cards use account-scoped sequential numbering (`assign_number` via `account.increment!(:cards_count)`) and route by number (`to_param` returns `number.to_s`).

### Authentication (More Sophisticated)

Three auth methods:
1. **Cookie sessions** — signed session token in cookie, same pattern as Campfire
2. **Bearer tokens** — `Identity#access_tokens` with per-token method permissions (`access_token.allows?(method)`)
3. **Magic links** — `MagicLinkMailer` sends sign-in links, creates `MagicLink` records
4. **Passkeys** — full WebAuthn via `has_passkeys` (Action Pack Passkeys gem). Challenge/creation endpoints under `/my/passkeys`

Account scoping happens before authentication — `require_account` runs first, then `require_authentication`. Multi-tenant URL routing uses `AccountSlug.encode(external_account_id)`.

### Authorization

Board-level access control via `Access` model (join table: user ↔ board with `involvement` enum: `access_only`, `watching`). Boards can be "all access" (everyone in the account) or restricted. When a card moves boards, inaccessible notifications get cleaned up async (`remove_inaccessible_notifications_later`).

Admin check: `Current.user.admin?` for account-level admin. Staff check: `Current.identity.staff?` for cross-account staff operations (SaaS tier).

### SaaS vs Self-Hosted

`Gemfile.saas` extends the base `Gemfile` with:
- **Queenbee** — 37signals' billing/subscription engine
- **console1984/audits1984** — production console auditing
- **Telemetry stack** — Sentry, Yabeda (Prometheus metrics), structured logging, GVL tools
- **Native push** — `action_push_native` for iOS/Android
- **ActiveResource** — for internal service communication

The `Account::MultiTenantable` concern has a class-level `multi_tenant` flag. When false (self-hosted), signups are only allowed when no accounts exist. When true (SaaS), open signups.

### Search Architecture (Dual-Engine)

This is the most interesting engineering in the codebase. `Search::Record` dynamically includes either `SQLite` or `Trilogy` module based on `connection.adapter_name`:

```ruby
class Search::Record < ApplicationRecord
  include const_get(connection.adapter_name)
```

**SQLite mode:** FTS5 virtual table with porter tokenizer. Upserts maintain a shadow FTS table alongside the main record. Uses `highlight()` and `snippet()` for result formatting.

**MySQL/Trilogy mode:** 16-shard FULLTEXT index. Records are distributed across `search_records_0` through `search_records_15` using CRC32 of account_id:
```ruby
SHARD_COUNT = 16
def shard_id_for_account(account_id)
  Zlib.crc32(account_id.to_s) % SHARD_COUNT
end
```
Each shard is a dynamically-generated class with its own table name. The `account_key` column (like `"account123"`) is indexed alongside content for tenant isolation in boolean mode queries. Content gets stemmed before storage.

### "Entropy" System

Unique concept — `Entropy` controls automatic card postponement. Each board (or account-level default) has an `auto_postpone_period` (3/7/30/90/365 days). Cards that go stale get auto-postponed. Board `Entropy` inherits from account `Entropy` if not set.

### Activity Spikes

`Card::ActivitySpike` detection job — monitors cards for unusual activity levels. Spikes are tracked per-card and surface in the UI.

### Events & Webhooks

Polymorphic `Event` model tracks all activity (`Eventable` concern). Events auto-dispatch to board webhooks via `Event::WebhookDispatchJob`. Webhook delivery tracking includes delinquency monitoring (`webhook_delinquency_trackers` table).

### Easter Egg: Drag & Strum

`drag_and_strum_controller.js` plays musical notes (vibes, banjo, harpsichord, mandolin, piano) when you drag cards across columns. Instruments are randomly selected. Classic 37signals whimsy.

### JavaScript

62 Stimulus controllers. Notable patterns:
- `drag_and_drop_controller.js` — full kanban drag-drop with CSS variable manipulation for column colors
- `combobox_controller.js`, `multi_selection_combobox_controller.js` — custom select components
- `beacon_controller.js` — likely for analytics/activity tracking
- `navigable_list_controller.js` — keyboard navigation
- `filter_form_controller.js` — dynamic filter UI

### Background Jobs

20 job classes. Key ones:
- `Card::ActivitySpike::DetectionJob` — analyzes card activity patterns
- `Account::DataImportJob` / `DataExportJob` — full data portability
- `Account::IncinerateDueJob` — GDPR-style data destruction after cancellation
- `Board::CleanInaccessibleDataJob` — permission-aware cleanup
- `Notification::Bundle::DeliverAllJob` / `DeliverJob` — batched email digests
- `Storage::MaterializeJob` / `ReconcileJob` — storage usage tracking
- `Webhook::DeliveryJob` — async webhook delivery

### SSRF Protection

`SsrfProtection` module with:
- Explicit DNS resolution via Cloudflare (1.1.1.1) and Google (8.8.8.8) nameservers
- Blocks private, loopback, link-local, ipv4-mapped, ipv4-compat ranges
- Additionally blocks carrier-grade NAT (100.64.0.0/10) and benchmark testing (198.18.0.0/15)

### CSS

64 hand-written CSS files. No Tailwind, no framework.

---

## 3. Writebook (Book Publishing)

### Domain Model

Single-tenant (same `Account.first` pattern as Campfire). Lean — 10 tables:

```
Account → Books → Leaves (delegated_type → Page | Section | Picture)
                → Accesses (reader/editor per user per book)
Users → Sessions
Leaves → Edits (version history, also delegated_type → Page | Section | Picture)
```

**Delegated type for content** — `Leaf` is the join model. Each leaf delegates to a `leafable` (Page, Section, or Picture). This is textbook `delegated_type` usage:
```ruby
delegated_type :leafable, types: Leafable::TYPES, dependent: :destroy
```

**Pages** store Markdown (not Action Text rich text — a custom `has_markdown` extension). **Sections** are chapter dividers with optional body text. **Pictures** are image-only with optional captions.

### Custom Markdown System

The most architecturally interesting part. Writebook extends Action Text with a parallel Markdown storage system:

**`ActionText::Markdown`** — a new Active Record model (`action_text_markdowns` table) that stores raw Markdown content. Renders HTML via Redcarpet with configurable extensions (autolink, fenced code blocks, strikethrough, tables, etc.).

**`ActionText::HasMarkdown`** — concern that adds `has_markdown :body` to models (mirrors `has_rich_text` API). Generates getter/setter/predicate methods via `class_eval`.

**Custom renderer** (`MarkdownRenderer`) adds:
- Auto-generated heading IDs with deduplication (second `## Intro` becomes `intro-2`)
- Anchor links on every heading (`<a href='#id' class='heading__link'>#</a>`)
- Images wrapped in lightbox-compatible links with download disposition

**Upload integration** — `ActiveStorage::Sluggable` gives attachments human-readable URL slugs instead of blob keys. Uploads get routes like `/u/my-image-abc123.png`.

### Version History (Edit System)

`Leaf::Editable` implements version tracking with a minimum 10-minute gap between versions:
```ruby
MINIMUM_TIME_BETWEEN_VERSIONS = 10.minutes
```

When editing:
1. Checks if enough time has passed since last edit
2. If yes: duplicates the current leafable (including attachments), creates an `Edit` record pointing to the old version, swaps the leaf to point to the new version
3. If no: just updates in place, touches the last edit timestamp

**Edit types:** `revision` (content change) and `trash` (moved to trash). The `dup_leafable_with_attachments` method properly handles Active Storage attachment cloning.

### Positioning System

`Positionable` concern uses fractional scoring:
- Each leaf has a `position_score` (float)
- Insert between items: `gap = (after - before) / (count + 1)`
- When gap drops below `1e-10`, triggers a rebalance (window function reassigns sequential integers):
```ruby
def rebalance_positions
  ordered = all_positioned_siblings.select("row_number() over (order by position_score, id) as new_score, id")
  sql = "update #{self.class.table_name} set position_score = new_score from (#{ordered.to_sql}) as ordered..."
  self.class.connection.execute sql
end
```
Uses pessimistic locking (`with_lock`) during position changes to prevent races.

### Book Access Control

Two-level enum: `reader` and `editor`. Books can have `everyone_access` (all users get reader access automatically on user creation). Administrators bypass access checks. The `update_access` method uses `upsert_all` for bulk permission updates — sets editors and readers in one shot, deletes removed users.

### JavaScript

24 Stimulus controllers. The standout is `arrangement_controller.js` (300+ lines):
- Full drag-and-drop AND keyboard-based reordering
- Shift-click multi-selection
- Move mode (Enter to start, arrows to move, Enter to confirm, Escape to cancel)
- Visual drag layer (clones DOM elements, positions them absolutely for smooth animation)
- Supports both creating new items via drag and reordering existing ones
- Custom data transfer types: `x-writebook/create` and `x-writebook/move`

`reading_progress_controller.js` tracks book reading position in localStorage and marks the last-read leaf.

### Search

SQLite FTS5 on leaves, similar to Campfire but with extras:
- BM25 ranking with title weighted 2x: `order(Arel.sql("bm25(leaf_search_index, 2.0)"))`
- Query sanitization: strips invalid FTS characters, removes unbalanced quotes
- Highlight extraction for search result display with `<mark>` tags
- `matches_for_highlight` extracts unique matching terms from content

### Slug System

Books and leaves get slugged URLs: `/:book_id/:book_slug/:leaf_id/:leaf_slug`. Direct routes use `constraints: { id: /\d+/ }` to differentiate from other routes. Books also respond to `.md` format for Markdown export.

### Background Jobs

Zero custom jobs. The simplest of the three — no async work at all. Everything is synchronous.

### CSS

26 hand-written CSS files. Notable: `house.css` (the markdown editor component), `arrangement.css` (drag-drop UI), `product.css` (public-facing book display).

---

## Cross-Project Patterns

### Shared Authentication DNA

Campfire and Writebook share nearly identical authentication code — same `Authentication` concern structure, same cookie strategy, same `Session` model. Fizzy evolved the pattern further with Identity separation, bearer tokens, magic links, and passkeys. This is clearly a progression — you can see the lineage.

### `Current` Object

All three use `ActiveSupport::CurrentAttributes`:
- **Campfire/Writebook:** `Current.session → Current.user`, `Current.account` is always `Account.first`
- **Fizzy:** `Current.session → Current.identity → Current.user` (account-scoped user lookup)

### Search Strategy

All three use SQLite FTS5 with manual SQL for index maintenance. Fizzy adds the MySQL sharded variant for SaaS scale. The pattern is consistent: `after_create_commit` / `after_update_commit` / `after_destroy_commit` callbacks sync a parallel search index.

### No Devise, No Pundit, No CanCanCan

All auth and authz is hand-rolled. Roles are simple enums. Permission checks are one-liners in concerns. This is a deliberate philosophical choice — fewer dependencies, more control, simpler to understand.

### Turbo Stream Broadcasts

Campfire and Fizzy use `broadcast_append_to` / `broadcast_remove_to` / `broadcast_prepend_later_to` for real-time updates. Writebook doesn't — it's a content publishing tool, not a collaborative real-time app.

### Concern-Heavy Architecture

All three projects lean heavily on concerns for code organization. Cards in Fizzy have 20+ concerns. Users in every project have 5-10 concerns. Models are thin shells that declare associations and mix in behavior. Controller concerns handle cross-cutting logic (auth, platform detection, version headers).

### Single-Tenant vs Multi-Tenant

Campfire and Writebook are designed for ONCE (buy once, self-host). They use `Account.first` — literally one account, no routing. Fizzy supports both modes via `multi_tenant` flag, with proper tenant isolation (account-scoped queries, access controls, URL-based account routing).

### No Heavy JS, No Build Step

All three use Importmap (no webpack/esbuild). Stimulus controllers handle all interactivity. The most complex JS is Campfire's message controller (scroll management, pagination, optimistic rendering) and Writebook's arrangement controller (drag-drop with keyboard support). Fizzy's `drag_and_strum_controller.js` is the most whimsical — plays music while you drag.

### Zero Active Job in Writebook

Campfire has 3 jobs. Fizzy has 20. Writebook has 0. This mirrors product complexity — a book publisher doesn't need async processing. Chat needs push notifications. A kanban board needs webhook delivery, notification bundling, data import/export, and storage tracking.

### CSS Philosophy

All three: hand-written CSS, component-based file organization, no utility framework. File counts: Campfire 26, Writebook 26, Fizzy 64. Custom CSS injection via `Account#custom_styles` in both Campfire and Writebook.

---

## 4. Views & Helpers — Deep Analysis

### View Architecture Overview

| Project | View files | Helper files | Turbo Stream templates | JSON templates | Format variants |
|---------|-----------|-------------|----------------------|----------------|----------------|
| Campfire | ~78 | 27 | 3 | 2 (jbuilder) | HTML, Turbo Stream, JSON, SVG |
| Fizzy | ~120+ | 30 | ~10 | ~12 (jbuilder) | HTML, Turbo Stream, JSON |
| Writebook | ~75 | 14 | 3 | 1 (jbuilder) | HTML, Turbo Stream, Markdown |

---

### Campfire Views

#### Layout Architecture

Single application layout (`layouts/application.html.erb`) with a `yield`-based slot system — `:nav`, `:sidebar`, `:footer`, `:head`. Views compose themselves by injecting content into these slots via `content_for`. The layout itself is minimal: skip-nav link, flash messages (animated self-removing via `element-removal` Stimulus controller), and a permanent sidebar `<aside>`.

Key pattern: the sidebar is a `turbo_frame_tag` with `turbo_permanent` — it survives Turbo navigation and only reloads via explicit frame src changes. This is how the sidebar persists across room switches without re-rendering.

```erb
<aside id="sidebar" data-controller="toggle-class" data-toggle-class-toggle-class="open">
  <%= yield :sidebar %>
</aside>
```

#### Message Rendering Pipeline

The most sophisticated view code. Messages go through a multi-layer rendering chain:

1. **`_message.html.erb`** — outer shell with `cache message` block, calls `message_tag` helper for the container div with all the Stimulus data attributes
2. **`_presentation.html.erb`** — just renders `message_presentation(message)` — dispatches to text, attachment, or sound rendering
3. **`_actions.html.erb`** — contextual action menu (boost reactions, reply, copy link, edit, download/share for attachments)
4. **`_template.html.erb`** — a `<script type="text/template">` used for optimistic client-side message rendering before server confirmation

The template partial is clever — it's a static HTML template with `$placeholder$` tokens that the `client_message.js` Stimulus controller fills in on the fly. This avoids a server round-trip for the sender's own messages.

Messages are fragment-cached at the model level. The `_message` partial uses `cached: true` in collection rendering — Rails' multi-key cache fetch for collections.

#### Turbo Stream Usage

Surgical — only 3 Turbo Stream templates:
- `create.turbo_stream.erb` — `turbo_stream.append` the new message to the room's messages div
- `destroy.turbo_stream.erb` — `turbo_stream.remove` the message
- `rooms/refreshes/show.turbo_stream.erb` — room refresh mechanism

Live updates come through Action Cable broadcasts, not Turbo Stream responses. The `turbo_stream_from @room, :messages` tag in the room view subscribes to the room's broadcast channel.

#### SVG Avatar Generation

`users/avatars/show.svg.erb` — server-side SVG generation for user avatars. Uses CRC32 hash of user ID to deterministically pick from 18 hand-chosen colors. Renders initials with responsive `textLength` when 3+ characters. No external avatar service, no image generation — just computed SVGs. Clean.

#### Sidebar Architecture

`users/sidebars/show.html.erb` is the sidebar's main view — a permanent Turbo Frame that contains both direct messages (horizontally scrolled) and shared rooms (vertically listed). Each room uses a `sorted-list` Stimulus controller for client-side reordering when new messages arrive. Direct rooms show avatar groups with up to 4 member avatars stacked.

The sidebar subscribes to two Turbo Streams: `:rooms` (global room updates) and `Current.user, :rooms` (per-user unread updates).

---

### Campfire Helpers

#### Data Attribute Factories

The dominant helper pattern is building complex Stimulus `data-*` attribute hashes. Nearly every helper returns a tag with 5-15 data attributes wiring up controllers, actions, targets, and outlets.

`MessagesHelper#message_area_tag` is the best example — a single `tag.div` call with nested controller declarations for `messages`, `presence`, and `drop-target`, plus event action strings concatenated from private methods:

```ruby
def message_area_tag(room, &)
  tag.div id: "message-area", data: {
    controller: "messages presence drop-target",
    action: [ messages_actions, drop_target_actions, presence_actions ].join(" "),
    messages_first_of_day_class: "message--first-of-day",
    # ... 6 more data attributes
  }, &
end
```

The action strings themselves are built by concatenating multiple private methods that each return a fragment of Stimulus action notation. This keeps the ERB clean — views call `message_area_tag(@room)` and get a fully-wired container.

#### Content Filter Pipeline

`ContentFilters` module defines a `TextMessagePresentationFilters` pipeline — three filter classes applied in sequence to Action Text content:

1. **`RemoveSoloUnfurledLinkText`** — when a message is just a single URL that got unfurled (OpenGraph embed), strips the raw URL text, leaving only the embed
2. **`StyleUnfurledTwitterAvatars`** — detects Twitter/X avatar URLs in OG embeds and adds a CSS class for special styling
3. **`SanitizeTags`** — whitelist-based HTML tag filter (allows 30+ standard tags, strips everything else)

The Twitter filter even handles x.com → twitter.com normalization with a domain mapping hash.

#### Attachment Presentation Object

`Messages::AttachmentPresentation` is a full presentation object (not a helper module) — instantiated with a message and view context, delegates `tag`, `link_to`, etc. to the context. Handles:
- Image attachments → lightboxed preview with aspect-ratio-aware sizing
- Video attachments → `<video>` tag with poster frame from preview
- Other files → icon + filename + download link + Web Share API button

Dimension calculations scale down to `THUMBNAIL_MAX_WIDTH/HEIGHT` while preserving aspect ratio — proper math with scale factors.

#### Translation System

`TranslationsHelper` is delightfully handcrafted — a `TRANSLATIONS` hash mapping keys to emoji-flagged language variants (🇺🇸, 🇪🇸, 🇫🇷, 🇮🇳, 🇩🇪, 🇧🇷, 🇯🇵). Renders as a `<details>` popup with a globe icon. No i18n gem, no YAML locale files — just a Ruby hash with emoji flags. Covers 7 languages for ~12 UI strings. This is the "ONCE" philosophy: ship something simple that works for the 80% case.

---

### Fizzy Views

#### Layout Architecture

More structured than Campfire. The layout has a proper `<header>` with mobile action stacking, shared partials for flash messages and timezone detection, and a persistent footer with three permanent frames: the bottom bar, pins tray, and notifications tray.

```erb
<div id="footer_frames" data-turbo-permanent="true">
  <%= render "bar/bar" %>
  <%= render "my/pins/tray" %>
  <%= render "notifications/tray" %>
</div>
```

The body tag is a Stimulus controller party — `local-time`, `timezone-cookie`, `turbo-navigation`, `theme`, plus four `bridge--*` controllers for native iOS/Android integration. The `data-bridge-platform` and `data-bridge-components` attributes enable conditional behavior based on whether the user is in a native app wrapper.

#### Board/Column/Card View Hierarchy

The deepest partial nesting in all three projects. The board show page (`boards/show.html.erb`) cascades through:

```
boards/show.html.erb
  └─ boards/show/_columns.html.erb (the card-columns container with drag-and-drop)
       ├─ boards/show/_not_now.html.erb
       ├─ boards/show/_stream.html.erb
       ├─ boards/show/_column.html.erb (per column, cached)
       │    └─ column_frame_tag → boards/columns/show.html.erb (lazy loaded via Turbo Frame)
       ├─ boards/show/_closed.html.erb
       └─ boards/show/menu/_columns.html.erb
```

Columns are lazy-loaded — the board page renders column shells with headers, then each column's card list is a Turbo Frame with a `src` attribute that loads asynchronously. This means the board page renders instantly with column headers, then cards stream in per-column.

#### Card Display Variants

Cards have 4 distinct display modes, each with its own partial subtree:

- **`display/preview/`** — compact card in kanban columns (title, tags, assignees, boosts, comments count)
- **`display/perma/`** — full card detail view (everything: board link, tags, steps, reactions, background)
- **`display/mini/`** — minimal card in lists (assignees, meta, tags)
- **`display/common/`** — shared partials across modes (stamp, background, assignees)

Each mode assembles from granular sub-partials. The preview card alone renders 10 partials: board, tags, steps, attachment indicator, column name, columns picker, stamp, meta, boosts, comments, bubble. This is extreme composition — no partial does more than one thing.

#### Turbo Morphing Integration

Fizzy uses Rails 8's Turbo morphing extensively. The `column_frame_tag` helper includes:

```ruby
data: {
  action: "turbo:before-frame-render->frame#morphRender turbo:before-morph-element->frame#morphReload"
}
```

And when a `src` is present, sets `refresh: :morph` on the frame. This means frame reloads use DOM morphing instead of replacement — smoother updates that preserve scroll position and focus state within columns.

The collapsible columns controller listens for `turbo:before-morph-attribute` to prevent morphing from toggling collapsed state.

#### JSON API Layer

Fizzy has a proper JSON API via jbuilder templates — `boards/_board.json.jbuilder`, `cards/_card.json.jbuilder`, `comments/index.json.jbuilder`, etc. This supports the native mobile apps (iOS/Android) that use the same Rails backend. Campfire and Writebook have minimal-to-no JSON rendering.

#### Native Bridge Integration

Views are littered with `bridge--*` Stimulus controller references:
- `bridge--buttons` — surfaces action buttons to native toolbar
- `bridge--overflow-menu` — populates native overflow menu
- `bridge--form` — hooks form submission into native loading indicators
- `bridge--title` — syncs page title to native nav bar
- `bridge--share` — triggers native share sheet

Hidden buttons with `bridge__overflow_menu_target: "item"` attributes are invisible in the web UI but get picked up by the native bridge to populate native menus. This is Turbo Native / Strada in action.

---

### Fizzy Helpers

#### The Icon System

`ApplicationHelper#icon_tag` renders `<span class="icon icon--{name}">` — a CSS-based icon system. No inline SVGs, no icon fonts — just semantic class names that map to CSS backgrounds or masks. Used everywhere: `icon_tag("arrow-left")`, `icon_tag("settings")`, `icon_tag("notification-bell-reverse-access_only")`. The helper is 3 lines but appears 100+ times across views.

#### Column Helper — View Component Pattern

`ColumnsHelper#column_tag` is the most complex helper in any of the three projects. It builds a full kanban column `<section>` with:
- Collapsed/expanded state classes
- Drag-and-drop target attributes
- Navigable list integration
- CSS variable injection for column colors
- Nested transition container with its own set of controllers

This is effectively a view component implemented as a helper — it takes a block, yields into a structured container with 15+ data attributes. The `column_frame_tag` companion creates morph-enabled Turbo Frames for each column.

#### Pagination System

`PaginationHelper` is a full pagination framework — 80+ lines implementing:
- Manual pagination (click "Load more")
- Automatic pagination (scroll-triggered via `fetch-on-visible` controller)
- Day-based timeline pagination (for activity feeds with date grouping)
- Frame-based page loading (each page is a Turbo Frame)
- Pagination link state management

Two public entry points: `with_manual_pagination` and `with_automatic_pagination` — each wraps a block in the right frame structure and appends the appropriate next-page link.

#### Notification Helpers

`NotificationsHelper` contains a complete notification rendering system — translates raw `Event` model data into human-readable titles and bodies:

```ruby
when "card_assigned" then "Assigned to #{event.assignees.pluck(:name).to_sentence}"
when "card_auto_postponed" then "Moved to Not Now due to inactivity"
when "comment_created" then comment.body.to_plain_text.truncate(200)
```

Handles edge cases: if a `card_published` event was self-assigned, it maps to `card_assigned` action instead. This is presentation logic that would live in a decorator/presenter in many codebases — here it's a helper.

#### Bridge Helper — Native Integration

`BridgeHelper` generates data for native app integration: icon URLs via asset pipeline, hidden buttons that the Strada bridge picks up, share descriptions computed from card/board metadata. The `bridged_share_url_button` creates an invisible button that native apps detect and wire to the platform share sheet.

#### Filter UI Framework

`FiltersHelper` is essentially a DSL for building filter UIs — `filter_chip_tag`, `filter_dialog`, `filter_title`, `filter_place_menu_item`, `collapsible_nav_section`, `filter_hotkey_link`. Each generates the right combination of Stimulus controllers, ARIA attributes, and CSS classes for an accessible, keyboard-navigable filter panel.

#### Form Helpers Pattern

Both `auto_submit_form_with` and `bridged_form_with` wrap `form_with` to inject additional Stimulus controllers. `bridged_form_with` adds `bridge--form` controller with submit start/end actions for native loading indicators. This composition pattern avoids touching `form_with` internals — just decorating the data hash.

#### Rich Text Prompts

`RichTextHelper` generates custom HTML elements (`<lexxy-prompt>`) for autocomplete triggers — `@` for mentions, `#` for tags/cards. These wire into a custom rich text editor component (Lexxy, not Trix) with configurable triggers, remote filtering, and space-in-search support.

---

### Writebook Views

#### Layout Architecture

The simplest layout. Five yield slots: `:head`, `:header`, `:toolbar`, `:sidebar`, `:footer`. No permanent frames, no Action Cable subscriptions. The body has three controllers: `fullscreen`, `lightbox`, `touch`.

The `#toolbar` slot is unique to Writebook — it hosts the editing toolbar (save button, edit/read mode toggle, history viewer, House markdown toolbar) when editing a page.

#### Content Type Rendering

The `leafables/show.html.erb` template is a clean delegated-type dispatcher:

```erb
<% if @leaf.section? %>
  <div class="page--section"><h1>...</h1></div>
<% elsif @leaf.page? %>
  <div class="page--page"><%= sanitize_content(@leaf.page.body.to_html) %></div>
<% elsif @leaf.picture? %>
  <figure class="page--picture">...</figure>
<% end %>
```

No polymorphic partial rendering, no complex dispatch — just three if branches. Each content type gets its own CSS class (`page--section`, `page--page`, `page--picture`) and its own rendering approach. Sections render via `simple_format`. Pages sanitize HTML through a custom `HtmlScrubber`. Pictures show an image with lightbox + caption.

#### Markdown Export Format

Writebook is the only project with non-HTML content format variants. Both books and leafables respond to `.md` format:

```erb
<%# books/show.md.erb %>
---
title: "<%= @book.title %>"
author: "<%= @book.author %>"
url: "<%= book_slug_url(@book) %>"
---
<%= raw @book.markable %>
```

YAML frontmatter + raw markdown body. This enables "export as markdown" — you can fetch any page or entire book as a `.md` file. The `markable` method on models returns the raw markdown content.

#### Table of Contents as Drag-Drop Zone

The book show page's table of contents doubles as a drag-and-drop arrangement surface. Each leaf is wrapped in `leaf_item_tag` which adds arrangement data attributes. The `arrangement_actions` helper generates 13 event bindings covering mouse drag, keyboard arrows (up/down/left/right + shift variants), space for toggle, enter for confirm, escape for cancel.

The leaf partial has dual-mode links — `hide_from_reading_mode` and `hide_from_edit_mode` CSS classes, controlled by a `data-controller="edit-mode"` toggle. When in edit mode, TOC links go to edit pages; in read mode, they go to the reading view.

#### Autosave Form Pattern

Writebook's editing forms use an `autosave` Stimulus controller wired via the `leafable_edit_form` helper:

```ruby
data: {
  controller: "autosave",
  action: "autosave#submit:prevent input@document->autosave#change house-md:change->autosave#change",
  autosave_clean_class: "clean",
  autosave_dirty_class: "dirty",
  autosave_saving_class: "saving"
}
```

Three visual states (clean/dirty/saving) communicated via CSS classes. Listens for both standard `input` events and custom `house-md:change` events from the markdown editor.

#### Custom Turbo Stream Actions

`TurboStreamActionsHelper` extends Turbo Streams with a custom `scroll_into_view` action:

```ruby
def scroll_into_view(id, animation: nil)
  turbo_stream_action_tag :scroll_into_view, target: id, animation: animation
end
```

Used in `leafables/create.turbo_stream.erb` with `animation: :wiggle` — when a new leaf is created, it scrolls into view and wiggles. The module is prepended to `Turbo::Streams::TagBuilder` so it works in any Turbo Stream context.

#### Concurrent Editing Indicator

`leaves/_being_edited_by.turbo_stream.erb` and `_being_edited_indicator.html.erb` — when another user is editing a leaf, a banner appears via Turbo Stream broadcast. Light presence without the full Action Cable channel complexity of Campfire.

---

### Writebook Helpers

#### Arrangement Helper

`ArrangementHelper` generates the data attributes for the drag-and-drop book arrangement system. The `arrangement_actions` method builds 13 event→action mappings as a hash, then joins them. This is the keyboard-accessible drag-and-drop system — arrow keys move items, space toggles move mode, enter confirms, escape cancels.

#### Book Navigation Helpers

`BooksHelper` has a complete navigation system — `link_to_previous_leafable`, `link_to_next_leafable`, `link_to_first_leafable`. Each generates hotkey-enabled navigation links (`keydown.left` / `keydown.right` / `touch:swipe-left` / `touch:swipe-right`). Navigation wraps — after the last leaf, "next" returns to the table of contents.

The `book_part_create_button` helper creates buttons that double as drag sources — you can click to create a new page/section/picture, or drag the button into the TOC to insert at a specific position. The `draggable: true` and `dragstart/dragend` actions enable this dual behavior.

#### Search Highlighting

`SearchesHelper#highlight_searched_content` takes search terms, converts them to whole-word regex matchers (`/\bterm\b/`), and applies Rails' `highlight` helper with `sanitize: false` (content is pre-sanitized via `HtmlScrubber`). The FTS5 match extraction from `LeavesHelper` provides the terms.

#### Page Word Counter

`PagesHelper#word_count` — splits content on whitespace, counts, formats with `number_with_delimiter` and `pluralize`. Simple, but it appears in the TOC next to each page and in the page editor toolbar.

#### Hide From User

`ApplicationHelper#hide_from_user_style_tag` injects a `<style>` tag that hides any element with `data-hide-from-user-id="#{Current.user.id}"`. This is used for concurrent editing indicators — you don't see your own "being edited" banner.

---

### Cross-Project View & Helper Patterns

#### Shared Helper DNA

These helpers appear (with slight variations) across multiple projects:

| Helper | Campfire | Fizzy | Writebook |
|--------|----------|-------|-----------|
| `page_title_tag` | ✓ | ✓ (adds account name for multi-tenant) | uses `content_for(:title)` |
| `custom_styles_tag` | ✓ | — | ✓ |
| `auto_submit_form_with` | ✓ | ✓ | ✓ |
| `button_to_copy_to_clipboard` | ✓ | ✓ | ✓ (in InvitationsHelper) |
| `avatar_tag` / avatar rendering | ✓ (SVG-based) | ✓ (image-based with fallback) | — |
| `local_datetime_tag` | ✓ (ISO8601 datetime) | ✓ (Unix timestamp, morph-aware) | — |
| `EmojiHelper::REACTIONS` | ✓ (8 reactions) | ✓ (48 reactions) | — |
| `TranslationsHelper` | ✓ (7 languages, 12 strings) | — | ✓ (6 languages, 12 strings) |
| `version_badge` | ✓ | — | ✓ |
| `QrCodeHelper` | ✓ | ✓ | ✓ (in InvitationsHelper) |

The evolution is visible: Campfire's `local_datetime_tag` uses ISO8601, Fizzy's uses Unix timestamps and adds morph-refresh awareness. Campfire's emoji set is 8 reactions, Fizzy expanded to 48. The `TranslationsHelper` is nearly identical between Campfire and Writebook (ONCE products) but absent from Fizzy (SaaS, presumably using proper i18n).

#### Helper Complexity Scaling

- **Writebook** (14 helpers) — focused on content rendering, arrangement, and navigation. No real-time concerns, no notification systems. The most complex helper is `ArrangementHelper` at ~30 lines.
- **Campfire** (27 helpers) — adds real-time data attribute factories, content filtering pipeline, attachment presentation object. `MessagesHelper` and the sidebar helper are the heaviest.
- **Fizzy** (30 helpers) — adds pagination framework, notification system, filter DSL, native bridge integration, column/card composition. `ColumnsHelper`, `PaginationHelper`, and `FiltersHelper` are each 50-80+ lines of sophisticated UI construction.

#### The Data Attribute Factory Pattern

All three projects use helpers primarily as **Stimulus data attribute factories** — the main job of most helpers is constructing the right combination of `data-controller`, `data-action`, `data-*-target`, `data-*-value`, and `data-*-class` attributes. Views stay clean because the complex wiring lives in helpers.

Campfire concatenates action strings from private methods. Fizzy uses `token_list` for conditional class/action composition. Writebook builds action hashes then joins them.

#### View-Level Caching Strategy

- **Campfire**: Fragment caching on messages (`cache message`), sidebar direct room memberships (`cache membership`). Collection rendering with `cached: true`.
- **Fizzy**: Fragment caching on card containers (`cache card`), column partials (cached with custom key lambdas: `cached: ->(column){ [column, column.leftmost?, column.rightmost?] }`).
- **Writebook**: Fragment caching on book show page (`cache [@book, @book.editable?]`), leaf partials (`cache [leaf, leaf.book, leaf.book.editable?]`), library page (`cache [@books, signed_in?]`).

All three cache at the partial level with model-based keys. Fizzy's lambda-based cache keys for columns are the most sophisticated — including positional information in the key.

#### No Presenters, No View Components, No Decorators

Zero. All three projects put presentation logic directly in helpers and partials. No `draper`, no `view_component`, no `phlex`. The closest thing to a presenter is Campfire's `Messages::AttachmentPresentation` class — and even that's a one-off in the helpers directory, not a pattern.

This is a deliberate architectural choice: keep the view layer as close to "plain Rails" as possible. Helpers are the composition mechanism. Partials are the reuse mechanism. Nothing else.

#### Turbo Stream Minimalism

Despite being real-time apps, the Turbo Stream template count is remarkably low. Campfire has 3, Writebook has 3, Fizzy has ~10. Most real-time updates flow through Action Cable broadcasts using `broadcast_append_to` / `broadcast_remove_to` in models — not through controller-rendered Turbo Stream templates.

The Turbo Stream templates that do exist handle form submission responses (create/update/destroy) where the server needs to tell the client exactly what DOM operation to perform. Fizzy uses more because it has more interactive operations (column create/update, closure create/destroy, comment CRUD).

#### Format Variant Strategy

- **Campfire**: HTML + Turbo Stream + JSON (jbuilder for autocomplete) + SVG (avatar generation)
- **Fizzy**: HTML + Turbo Stream + JSON (jbuilder for API/native apps) — the most formats because it serves web + native
- **Writebook**: HTML + Turbo Stream + Markdown — the only project with a content export format

Writebook's `.md.erb` templates are unique — YAML frontmatter rendered by ERB, followed by raw markdown content. This enables the "download as markdown" feature for any page or entire book.
