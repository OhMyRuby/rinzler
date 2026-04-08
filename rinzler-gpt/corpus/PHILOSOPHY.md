# 37signals Coding Philosophy

Assembled from dev.37signals.com. Seventeen articles. One coherent philosophy.

---

## I. The Central Doctrine: Vanilla Rails Is Enough

The throughline in everything they write is this: **resist complexity by default**. Their stack is intentionally boring.

- **No React, no Vue.** Server-side rendering with ERB, Hotwire for interactivity.
- **No Redis** (for most things). Replaced by `solid_cache`, `solid_queue`, `solid_cable` — all backed by the relational database.
- **No JavaScript bundlers.** Import maps + Propshaft. CSS served directly. They call it `#nobuild`.
- **No service objects / interactors / command pattern layers** between controllers and models. Controllers call models. That's the stack.
- **No React Native.** Hotwire Native for mobile apps. One codebase, HTML over the wire.
- **Minitest with fixtures**, not RSpec or factories.
- **Kamal** for deployment. No Kubernetes.

The philosophy is explicit: *fewer dependencies mean fewer future headaches*. A vanilla stack stays nimble across years and upgrades.

---

## II. Domain Modeling: Rich, Expressive, and Bold

### Rich Domain Models Over Service Objects

Business logic lives on models — not in service objects that merely proxy to models. Controllers call expressive public methods directly:

```ruby
@contact.designate_to(@box)
recording.incinerate
recording.copy_to(destination_bucket)
```

When a model method needs internal complexity, it delegates to a dedicated class behind a clean API:

```ruby
def incinerate
  Incineration.new(self).run
end
```

The public API is domain-fluent. The implementation detail is hidden. Single Responsibility is about *interfaces*, not *class sizes*.

### Domain-Driven Naming: Be Bold

They explicitly reject generic, sterile naming. Code should reflect the real domain vocabulary — even when that vocabulary is unusual:

```ruby
module Person::Tombstonable
  def decease
    erect_tombstone
    remove_administratorships
    remove_accesses_later
  end
end
```

vs. a generic `deactivate` or `soft_delete`. The naming carries meaning. HEY has `Examiner`, `Petitioner`, `clearance_petitions` — language borrowed directly from how users think about their workflow.

> "Don't be aseptic; double down on boldness."

### The Fractal Principle

Good code repeats the same qualities at every abstraction level: domain-driven language, encapsulation, cohesion, and consistent abstraction level (no mixing high-level orchestration with low-level implementation in the same method). They call this "fractal code."

```ruby
# High level: clear orchestration
def relay_now
  relay_to_or_revoke_from_timeline
  relay_to_webhooks_later
end

# Low level: focused implementation
class Timeline::Relayer
  def relay
    record
    broadcast
  end
end
```

---

## III. Concerns: A Pattern They Actually Defend

Rails concerns have a bad reputation. 37signals disagrees — conditionally.

**The rule:** A concern must represent a genuine domain trait with "has trait" or "acts as" semantics. Not arbitrary code-splitting. Not a junk drawer.

```ruby
# app/models/recording.rb
class Recording < ApplicationRecord
  include Completable
  include Incineratable
  include Copyable
end

# app/models/recording/completable.rb  (model-specific → lives in subdirectory)
module Recording::Completable
  extend ActiveSupport::Concern
end
```

**Model-specific concerns** go in `app/models/<model_name>/` to avoid namespace pollution. **Shared concerns** go in `app/models/concerns/`.

Concerns provide the clean public API; service classes handle the complexity underneath:

```ruby
module Account::Closable
  def terminate
    purge_or_incinerate if terminable?
  end

  private
    def purge
      Account::Closing::Purging.new(self).run
    end
end
```

---

## IV. The Delegated Type Pattern

Their crown jewel for content-heavy systems. Used throughout Basecamp and HEY.

**Problem:** Many similar-but-different content types (messages, documents, comments, uploads) that share operations but have different schemas. STI bloats tables. Polymorphic associations invert the relationship awkwardly.

**Solution:**

```
Recording (parent, lean — metadata only, never changes schema)
  └── Recordable (delegated type: Message | Document | Comment | Upload)
```

- The `Recording` table is thin — timestamps, creator, references. Never modified for new types.
- Each recordable type has its own table with type-specific columns.
- Recordables are **immutable** — new versions create new records. Enables efficient copying and version history.
- Controllers, views, and jobs work with recordings generically. Adding a new content type requires zero changes to shared infrastructure.

```ruby
class Recording < ApplicationRecord
  delegated_type :recordable
end

recordings.messages       # scope to type
recording.commentable?    # delegates to recordable
```

Basecamp has run this architecture for 10+ years without a major rewrite. Billions of recording rows, still fast because the table is lean.

---

## V. Active Record: Embrace It, Don't Fight It

They explicitly reject the idea that persistence should be strictly separated from domain logic. Their argument: since database-powered applications inherently couple domain logic and persistence, enforcing separation is often fighting the grain.

Active Record does real work:

```ruby
has_many :entries, dependent: :destroy
has_many :addressed_contacts, -> { distinct }, through: :entries
scope :accessible_to, ->(contact) { not_deleted.joins(:accesses) }
```

STI and delegated types are used without apology. Scopes encapsulate query complexity. The complexity hiding happens through Ruby encapsulation, not architectural layers.

---

## VI. Pragmatic Use of Callbacks and Globals

This is where they diverge most sharply from mainstream Rails advice.

**Callbacks for secondary concerns:** Auditing, tracking, orthogonal lifecycle behavior. Not core business logic.

```ruby
module Bucketable
  included do
    after_create { create_bucket! account: account unless bucket.present? }
  end
end
```

**`CurrentAttributes` as request context:** Instead of threading authentication/session data through every method signature, `Current` provides thread-safe implicit context:

```ruby
class Project < ApplicationRecord
  belongs_to :creator, default: -> { Current.person }
end
```

**`suppress` for targeted exceptions:**
```ruby
Event.suppress do
  @destination_recording = destination_bucket.record(...)
end
```

Their principle: *evaluate tradeoffs rather than follow dogma*. "Sharp knives" are contextual tools. Callbacks are ideal for simple orthogonal behavior; factories are justified for complex flows. The decision is situational, not ideological.

---

## VII. Frontend: Hotwire, Vanilla CSS, Zero Build

### Hotwire Philosophy

Full-stack ownership by individual programmers. Pairs (one designer, one programmer) ship complete features in 6-week cycles. This only works if one programmer can handle the entire stack — Rails makes that possible; Hotwire keeps it that way.

- **Turbo Frames** for in-place CRUD with zero JavaScript
- **Turbo Streams** for surgical DOM updates after server-side operations
- **Stimulus** for focused client-side behavior (37 lines of JS for full drag-and-drop in one feature)

Progressive enhancement: layer Hotwire on top of existing behavior rather than rewriting. Legacy code is extended with HTML attributes — no new coupling, no rewrite.

### Modern CSS Without a Build Step

Campfire uses vanilla CSS with no preprocessor, no framework.

**OKLCH color space** for better color management:
```css
--lch-gray: 96% 0.005 96;
--color-border: oklch(var(--lch-gray));
--color-border-transparent: oklch(var(--lch-gray) / 0.5);
```

**Custom properties as a mini-API** for component variants — declared vs. fallback pattern:
```css
color: var(--btn-color, var(--color-text));
```

**CSS `:has()` replacing JavaScript and server-side code:**
```css
/* Grey out rooms with disabled toggles — no JS, no server logic */
.membership-item:has(.btn.invisible) {
  opacity: 0.5;
}
```

**Zero viewport breakpoints.** One layout breakpoint in `ch` units (character-based, not pixel-based). Pointer/hover media queries for touch vs. mouse behavior.

---

## VIII. Infrastructure: Imperative Over Declarative

After leaving the cloud and Kubernetes, they chose Chef and Kamal over Terraform and Helm.

**Why:** Imperative tools force you to show your work. When a 3am PagerDuty fires, you want to read a Chef recipe and know exactly what should be true — not debug Kubernetes state reconciliation magic.

Auto-scaling is rejected as unnecessary overhead. They manually resize VMs. The friction is intentional: it ensures the ops team understands system state.

**Chef V2 principles:**
- Reduce cookbook complexity and interdependencies
- Eliminate scattered node attributes
- Embrace repetition and inline configuration over templates
- Decouple apps from shared "one-size-fits-all" cookbooks

---

## IX. Testing Philosophy

They write tests **after** code. Not TDD.

Key beliefs:
- Mocking slow dependencies produces fast but expensive, ineffective test suites. **Test the real thing.**
- Testable code ≠ well-designed code. You can have perfectly fast, isolated unit tests on a terrible architecture.
- Tests serve their purpose without being driven by them.
- **Pending tests** are legitimate — mark something for future coverage without blocking.

Shorter feedback loops (infrastructure work especially) may justify writing tests earlier. The principle is pragmatism, not dogma.

---

## X. Code Review and Team Culture

### Pull Request Reviews: The Small Stuff Matters

Code reviews are about three things equally:

1. **Names** — renaming matters. Past-tense event handlers (`linkClicked`), precise verbs (`click` not `forward`). Naming is the primary tool of comprehension.
2. **Style** — Rails idioms are not cosmetic. `.ids` over `.pluck(:id)`. `5.days.ago` over manual datetime arithmetic. One-lining simple conditionals.
3. **Consistency** — what automation can't enforce is the "thick middle ground" of coding standards. Reviews establish and transmit that culture.

> "Computers don't execute big pictures, they execute lines of code."

### The Radiating Programmer

Individual contributors push information out rather than waiting to be pulled into meetings. No daily standups — instead, async written updates, Hill Chart progress, decisions shared with an invitation for non-blocking feedback. Roughly one to two hours per week. Done at day's end to protect deep work time.

### Constraints as a Feature

Fixed time, variable scope. Small teams. These aren't obstacles — they're the mechanism that forces good decisions. Unlimited resources remove the feedback that teaches priorities.

---

## The Architecture at a Glance

| Layer | Principle |
|---|---|
| Stack | Vanilla, minimal deps, own your infrastructure |
| Architecture | Rich models, no service layer, Active Record embraced |
| Domain modeling | Bold expressive naming, fractal quality, delegated types |
| Cross-cutting concerns | Concerns + callbacks + CurrentAttributes, used contextually |
| Frontend | Hotwire + vanilla CSS, full-stack ownership, no build |
| Infrastructure | Imperative, explicit, no magic state reconciliation |
| Testing | Real dependencies, pragmatic not dogmatic |
| Process | Async communication, constraints enforced structurally |

It's a coherent worldview, not a collection of tips. The willingness to use globals, callbacks, and blended persistence all flows from the same root: **pragmatic tradeoffs beat ideological purity**. The question is always whether a technique works in practice at scale — and they've got the 10-year Basecamp production receipts to back it up.
