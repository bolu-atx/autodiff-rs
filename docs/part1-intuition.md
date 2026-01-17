# Understanding Autodiff, Part 1: Thinking in Graphs

*How to see math as pictures and gradients as flowing water*

---

I've always found automatic differentiation a bit magical. You write some math, call `.backward()`, and somehow the computer figures out all the derivatives. For years I used it without really understanding it.

Then I sat down and implemented one from scratch. Turns out it's not magic at all — it's actually a beautiful idea that clicks once you see it the right way.

Let me show you what I learned.

## Forget Calculus Class (For Now)

When most of us learned calculus, we memorized rules:

- d/dx(x²) = 2x
- d/dx(sin x) = cos x
- The chain rule: d/dx(f(g(x))) = f'(g(x)) · g'(x)

These rules work great on paper. But here's the thing — computers don't manipulate symbols the way we do. They work with numbers and data structures.

So instead of asking "how do I differentiate this expression?", let's ask a different question:

**How do I represent a mathematical expression in a way that makes differentiation natural?**

## Math as a Picture

Consider this function:

```
f(x, y) = (x · y) + sin(x)
```

Nothing fancy — multiplication, addition, a sine. But let's draw it differently. Instead of one line of symbols, let's trace how the computation actually happens:

```mermaid
graph BT
    x["x"]
    y["y"]
    mul["multiply"]
    sin["sine"]
    add["add"]
    out["output"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add
    add --> out

    style x fill:#a8d5ff
    style y fill:#a8d5ff
    style out fill:#90EE90
```

Read this bottom-to-top. The inputs `x` and `y` flow upward. They get combined by operations. Eventually we get our output.

This is a **computation graph**. Every mathematical expression can be drawn this way.

Let me add the actual values. Say x = 2 and y = 3:

```mermaid
graph BT
    x["x = 2"]
    y["y = 3"]
    mul["multiply<br/>2 × 3 = 6"]
    sin["sine<br/>sin(2) ≈ 0.91"]
    add["add<br/>6 + 0.91 = 6.91"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add

    style x fill:#a8d5ff
    style y fill:#a8d5ff
    style add fill:#90EE90
```

Now you can literally *see* the computation happening. Values flow up from inputs to output.

## The Question We Actually Want to Answer

Here's what we care about in machine learning:

> If I wiggle the inputs a tiny bit, how much does the output wiggle?

That's what a derivative is. It's a measure of sensitivity.

For our function at x=2, y=3:
- If I increase x slightly, how much does the output change? (That's ∂f/∂x)
- If I increase y slightly, how much does the output change? (That's ∂f/∂y)

The graph view makes this intuitive. Wiggle x, and that wiggle propagates up through every node that depends on x.

## Following the Wiggle

Let's trace what happens when we wiggle x:

```mermaid
graph BT
    x["x = 2<br/>↑ wiggle here"]
    y["y = 3"]
    mul["multiply<br/>feels the wiggle"]
    sin["sine<br/>feels the wiggle"]
    add["add<br/>feels both wiggles"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add

    linkStyle 0 stroke:#ff6b6b,stroke-width:3px
    linkStyle 2 stroke:#ff6b6b,stroke-width:3px
    linkStyle 3 stroke:#ff6b6b,stroke-width:3px
    linkStyle 4 stroke:#ff6b6b,stroke-width:3px

    style x fill:#ffcccc
```

Notice something important: **x connects to the output through two different paths**.

1. x → multiply → add → output
2. x → sine → add → output

Both paths carry the wiggle. The total effect on the output is the *sum* of both contributions.

This is the key insight. In a graph, derivatives flow backward along edges, and when paths merge, contributions add up.

## Local Rules, Global Result

Here's what makes autodiff elegant. Each operation only needs to know one thing: **how sensitive is my output to each of my inputs?**

Let's write these down:

| Operation | If input wiggles by ε... | Output wiggles by... |
|-----------|--------------------------|---------------------|
| add(a, b) | a wiggles by ε | ε (just passes through) |
| add(a, b) | b wiggles by ε | ε (just passes through) |
| multiply(a, b) | a wiggles by ε | b · ε (scaled by the other input) |
| multiply(a, b) | b wiggles by ε | a · ε (scaled by the other input) |
| sin(a) | a wiggles by ε | cos(a) · ε |

These are just the derivatives you learned in calculus, but think of them as **local exchange rates** for wiggles.

## The Chain Rule as Plumbing

Now here's the beautiful part. To find the total sensitivity from input to output, we just:

1. Find all paths from input to output
2. Multiply the local exchange rates along each path
3. Add up the contributions from all paths

Let's do it for ∂f/∂x:

**Path 1: x → multiply → add**
- x → multiply: exchange rate = y = 3 (the other input to multiply)
- multiply → add: exchange rate = 1 (addition just passes through)
- Total for this path: 3 × 1 = 3

**Path 2: x → sin → add**
- x → sin: exchange rate = cos(2) ≈ -0.42
- sin → add: exchange rate = 1
- Total for this path: -0.42 × 1 = -0.42

**Grand total: 3 + (-0.42) = 2.58**

That's ∂f/∂x. We computed a derivative without doing any symbolic manipulation — just multiplying and adding numbers along paths in a graph.

```mermaid
graph BT
    x["x = 2"]
    y["y = 3"]
    mul["×<br/>local rate: y=3"]
    sin["sin<br/>local rate: cos(2)≈-0.42"]
    add["+<br/>local rate: 1, 1"]

    x -->|"×3"| mul
    y --> mul
    x -->|"×(-0.42)"| sin
    mul -->|"×1"| add
    sin -->|"×1"| add

    style x fill:#ffcccc
```

## Why Not Just Go Forward?

You might wonder: can't we just wiggle each input and measure what happens at the output?

Yes! That's called **forward mode** autodiff. For each input, you trace the wiggle forward through the graph.

But think about a neural network. It might have millions of input parameters. Forward mode would require millions of passes through the graph — one for each parameter.

Here's the trick: what if we went **backward** instead?

## Thinking Backward

Instead of asking "if I wiggle x, what happens to output?", ask:

> "If the output needed to change, how much would each input need to change?"

Start at the output. Its "sensitivity to itself" is 1 (trivially). Now work backward:

```mermaid
graph BT
    x["x<br/>sensitivity: ?"]
    y["y<br/>sensitivity: ?"]
    mul["×<br/>sensitivity: ?"]
    sin["sin<br/>sensitivity: ?"]
    add["+<br/>sensitivity: 1"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add

    style add fill:#90EE90
```

The add node has sensitivity 1. It passes this backward to both its inputs (because addition has local rate 1 in both directions):

```mermaid
graph BT
    x["x<br/>sensitivity: ?"]
    y["y<br/>sensitivity: ?"]
    mul["×<br/>sensitivity: 1"]
    sin["sin<br/>sensitivity: 1"]
    add["+<br/>sensitivity: 1 ✓"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add

    style add fill:#c8e6c9
    style mul fill:#fff9c4
    style sin fill:#fff9c4
```

Now multiply and sin both have sensitivity 1. They propagate backward using their local rates:

- From multiply: x gets 1 × y = 1 × 3 = 3, y gets 1 × x = 1 × 2 = 2
- From sin: x gets 1 × cos(2) ≈ -0.42

But wait — x receives from both multiply AND sin! We add them up:

```mermaid
graph BT
    x["x<br/>sensitivity: 3 + (-0.42) = 2.58 ✓"]
    y["y<br/>sensitivity: 2 ✓"]
    mul["×<br/>sensitivity: 1 ✓"]
    sin["sin<br/>sensitivity: 1 ✓"]
    add["+<br/>sensitivity: 1 ✓"]

    x --> mul
    y --> mul
    x --> sin
    mul --> add
    sin --> add

    style add fill:#c8e6c9
    style mul fill:#c8e6c9
    style sin fill:#c8e6c9
    style x fill:#e1bee7
    style y fill:#e1bee7
```

**One backward pass gave us *all* the derivatives.**

This is reverse-mode autodiff. It's why PyTorch can train a model with 175 billion parameters — one forward pass to compute the loss, one backward pass to get all 175 billion gradients.

## The Mental Model

Here's how I think about it now:

**Forward pass**: Values flow upward like water. Inputs combine, transform, eventually reach the output.

**Backward pass**: Sensitivity flows downward like... inverse water? Each node receives sensitivity from above, multiplies by its local rate, and passes it down.

```mermaid
graph BT
    subgraph "Forward: Values Flow Up"
        direction BT
        i1["inputs"] --> o1["operations"] --> out1["output"]
    end

    subgraph "Backward: Sensitivity Flows Down"
        direction TB
        out2["output<br/>sens = 1"] --> o2["operations"] --> i2["inputs<br/>sens = gradients!"]
    end
```

When paths split going forward, sensitivities add going backward. When paths merge going forward, sensitivities... well, they just follow their edges backward.

## What About More Complex Graphs?

The same principle scales to any computation:

```mermaid
graph BT
    x["x"]
    y["y"]
    a["a = x + y"]
    b["b = x - y"]
    c["c = a × b"]

    x --> a
    y --> a
    x --> b
    y --> b
    a --> c
    b --> c

    style x fill:#a8d5ff
    style y fill:#a8d5ff
    style c fill:#90EE90
```

This is `c = (x + y)(x - y) = x² - y²`. The graph has a diamond shape — x and y each flow through two different paths before rejoining.

Backward pass still works the same way:
1. Start at c with sensitivity 1
2. c passes to a: sensitivity × (value of b)
3. c passes to b: sensitivity × (value of a)
4. a and b pass to x and y, with appropriate signs
5. x and y sum their incoming sensitivities

The answers come out to ∂c/∂x = 2x and ∂c/∂y = -2y. Exactly what we'd get from calculus, but computed mechanically through the graph.

## The Punchline

Automatic differentiation isn't really about calculus. It's about:

1. **Representation**: Seeing computation as a graph
2. **Locality**: Each operation only knows its own derivative
3. **Composition**: The chain rule falls out naturally from graph traversal
4. **Direction**: Going backward lets us compute all gradients at once

That's the conceptual foundation. In [Part 2](./part2-implementation.md), we'll build this in Rust — turning these pictures into actual code.

---

*Next: [Part 2 - Implementing Autodiff in Rust](./part2-implementation.md)*
