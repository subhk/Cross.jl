const CROSS_BANNER = """

"""

function print_cross_header(; E, Pr, χ, lmax, Nr, use_kore, table_ref)
    println(CROSS_BANNER)
    println()
    println("Simulation settings:")
    @printf("  %-18s : %12.6e\n", "Ekman (E)", E)
    @printf("  %-18s : %12.6f\n", "Prandtl (Pr)", Pr)
    @printf("  %-18s : %12.6f\n", "Radius ratio (χ)", χ)
    @printf("  %-18s : %12d\n", "lmax", lmax)
    @printf("  %-18s : %12d\n", "Nr", Nr)
    println("  Kore weighting      : ", use_kore ? "enabled (r³ formulation)" : "disabled (standard)")
    println()
    println("Reference modes (m, Ra_table):")
    for (m, Ra) in table_ref
        @printf("  m = %-3d   Ra_table = %.6e\n", m, Ra)
    end
    println("--------------------------------------------------------------")
    println()
end
