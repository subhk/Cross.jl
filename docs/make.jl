using Documenter
using Cross

makedocs(
    sitename = "Cross.jl",
    authors  = "Subhajit Kar",
    modules  = [Cross],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://subhk.github.io/Cross.jl/stable",
        assets     = ["assets/cross.css"],
        collapselevel = 2,
        sidebar_sitename = false,
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting_started.md",
            "First Problem" => "problem_setup.md",
            "Examples" => "examples.md",
        ],
        "Analysis Modes" => [
            "Overview" => "analysis/index.md",
            "Onset Convection (No Mean Flow)" => "analysis/onset_convection.md",
            "Biglobal (Axisymmetric Mean Flow)" => "analysis/biglobal_stability.md",
            "Triglobal (Non-Axisymmetric Mean Flow)" => "analysis/triglobal_stability.md",
        ],
        "User Guide" => [
            "Basic States" => "basic_states.md",
            "Tri-Global Analysis" => "triglobal.md",
            "MHD Extension" => "mhd_extension.md",
            "MHD User Guide" => "mhd_user_guide.md",
        ],
        "Theory" => [
            "Mathematical Foundations" => "theory/mathematical_foundations.md",
            "Spectral Methods" => "theory/spectral_methods.md",
        ],
        "Reference" => [
            "API Reference" => "reference.md",
            "Codebase Structure" => "codebase_structure.md",
            "Migration Guide (v2.0)" => "migration-v2.md",
            "FAQ" => "faq.md",
        ],
    ],
    # First migration: keep the build green while content/docstrings are converted.
    # These are tightened (removed) in Task 7 once conversion is complete.
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/subhk/Cross.jl",
    devbranch = "main",
    push_preview = false,
)
