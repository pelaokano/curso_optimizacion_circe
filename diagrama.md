flowchart TD
    A[Power System Environment<br/>IEEE 30-Bus Network] --> B[System State Observation<br/>Voltage Profile V = {V₁, V₂, ..., Vₙ}]
    
    B --> C{Voltage Within<br/>Nominal Range?<br/>Vₘᵢₙ ≤ Vᵢ ≤ Vₘₐₓ}
    
    C -->|Yes| D[Calculate Positive Reward<br/>R = f(minimal voltage deviation)]
    C -->|No| E[Calculate Negative Reward<br/>R = f(large voltage deviation)]
    
    D --> F[Neural Network Agent<br/>Policy π(s)]
    E --> F
    
    F --> G[Action Selection Process]
    
    G --> H[Action Space A]
    H --> I[Generator Reactive Power<br/>Adjustment: ΔQ_gen]
    H --> J[Transformer Tap Position<br/>Adjustment: Δt_tap]
    
    I --> K[Apply Control Actions<br/>to Power System]
    J --> K
    
    K --> L[Power Flow Analysis<br/>Calculate New Voltage Profile]
    
    L --> M[Update System State<br/>s_{t+1} = V_new]
    
    M --> N[Training Algorithm Selection]
    
    N --> O[Deep Q-Learning<br/>DQL]
    N --> P[Genetic Algorithm<br/>GA]
    N --> Q[Particle Swarm Optimization<br/>PSO]
    
    O --> R[Q-Network Update<br/>Q(s,a) ← Q(s,a) + α[r + γmax Q(s',a') - Q(s,a)]]
    P --> S[Population Evolution<br/>Selection, Crossover, Mutation]
    Q --> T[Particle Position Update<br/>Velocity and Position Adjustment]
    
    R --> U{Convergence<br/>Criteria Met?}
    S --> U
    T --> U
    
    U -->|No| V[Continue Training<br/>Next Episode/Generation/Iteration]
    U -->|Yes| W[Optimal Control Policy<br/>π*(s)]
    
    V --> B
    
    W --> X[Deploy Trained Agent<br/>for Real-time Voltage Control]
    
    X --> Y[Performance Evaluation<br/>• Voltage Stability<br/>• Convergence Speed<br/>• Control Quality]
    
    style A fill:#e1f5fe
    style F fill:#f3e5f5
    style W fill:#e8f5e8
    style Y fill:#fff3e0
    style H fill:#fce4ec
    style N fill:#f1f8e9
