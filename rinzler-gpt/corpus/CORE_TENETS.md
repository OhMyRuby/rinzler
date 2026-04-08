# 37signals Core Tenets

---

## 1. Boring stack, interesting product
Minimize dependencies. Use what Rails gives you before reaching for gems, frameworks, or services. Every dependency is a future maintenance burden. Vanilla stays nimble.

## 2. Pragmatic tradeoffs beat ideological purity
Evaluate techniques by whether they work at scale, not whether they conform to doctrine. Callbacks, globals, blended persistence — all acceptable when applied with judgment. "Sharp knives" are contextual tools, not forbidden ones.

## 3. Business logic belongs on the model
No service objects, interactors, or command layers between controllers and models. Rich domain models with expressive public APIs. Complexity hides inside the model behind clean interfaces, not above it in a parallel class hierarchy.

## 4. Name things boldly
Code vocabulary should mirror domain vocabulary. `decease` and `erect_tombstone` over `soft_delete`. `clearance_petition` over `pending_request`. Sterile generic names are a form of imprecision.

## 5. Concerns are valid — when they represent real traits
A concern must capture a genuine "has trait" or "acts as" semantic. Not arbitrary code-splitting. Each concern should read like a domain role the object actually plays.

## 6. Fractal quality: same standards at every level
Encapsulation, cohesion, domain language, and consistent abstraction level should hold from the highest architectural layer down to individual method bodies. Good code looks the same at any zoom level.

## 7. Full-stack ownership by individuals
One programmer ships the whole feature — models, controllers, views, JavaScript, CSS. Hotwire exists to make this possible without a separate frontend team. Keep the stack simple enough for one person to hold in their head.

## 8. No build, no framework for CSS and JS
Vanilla CSS with modern platform features (`:has()`, OKLCH, nesting, custom properties). Import maps over bundlers. The web platform is powerful enough; build tools add friction without proportional value.

## 9. Imperative infrastructure over declarative magic
Write infrastructure code that shows its work. Explicit command sequences you can read and reason about at 3am beat state reconciliation systems that hide what's actually happening. Understand your system state; don't let a tool manage it for you.

## 10. Test the real thing
No mocks for slow dependencies — hit the real database, real services. Testable code is not the same as well-designed code. Write tests after the code. Pending tests are better than no tests.

## 11. Small stuff in reviews is not nitpicking
Names, idioms, and consistency are load-bearing. They compose into the team's shared understanding of what good code looks like. Reviews transmit culture that automation cannot enforce.

## 12. Constraints are the mechanism, not the obstacle
Fixed time, variable scope. Small teams. The friction of limited resources forces the prioritization decisions that unlimited resources defer indefinitely. Design your working environment so constraints are felt during execution, not just stated in planning.
