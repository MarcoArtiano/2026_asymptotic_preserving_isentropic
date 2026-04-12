using LinearAlgebra
using DoubleFloats
using Printf
using CairoMakie

struct DATA1D{RealT}
    DN::RealT  # Density
    VX::RealT  # x-velocity
end

DATA1D(RealT) = DATA1D(RealT(0.0), RealT(0.0))

struct Grid1D{RealT}
    x::RealT
end

struct SolverConfig1D{RealT,GeopotentialT}
    # Domain
    XMIN::RealT
    XMAX::RealT

    # Grid
    XNUM::Int
    NG::Int
    XNUM_TOT::Int
    dx::RealT

    # Physical parameters
    eps2::RealT
    # Time integration
    T::RealT
    CFL::RealT

    # Computed bounds
    IBEG::Int
    IEND::Int

    eps::RealT
    GAMMA::RealT
    GAMAM1::RealT
    GMBGMM1::RealT
    GMM1BGM::RealT
    OBGMM1::RealT
    si::RealT
    phi::GeopotentialT
    AP::Bool
    Reconstruct::Bool
    boundary_reflective::Bool
    viscosity::Bool
end

# IMEX-RK Tableau
struct IMEXTableau{RealT}
    stages::Int
    at::Matrix{RealT}
    a::Matrix{RealT}
    omegat::Vector{RealT}
    omega::Vector{RealT}
end

function create_config(
    xno,
    t_final,
    eps,
    GAMMA,
    si,
    phi,
    AP,
    Reconstruct,
    boundary_reflective,
    CFL,
    viscosity,
)
    # Default parameters matching the C code
    RealT = eltype(si)
    XMIN = RealT(0.0)
    XMAX = RealT(1.0)

    XNUM = xno
    NG = 2      # 2nd order two ghost cells on each side

    XNUM_TOT = XNUM + 2 * NG
    dx = RealT((XMAX - XMIN) / XNUM)


    T = RealT(t_final)

    IBEG = 1 + NG
    IEND = XNUM + NG
    eps2 = eps^2

    GAMAM1 = GAMMA - RealT(1.0)
    GMBGMM1 = GAMMA / GAMAM1
    GMM1BGM = GAMAM1 / GAMMA
    OBGMM1 = RealT(1.0) / GAMAM1
    return SolverConfig1D(
        XMIN,
        XMAX,
        XNUM,
        NG,
        XNUM_TOT,
        dx,
        eps2,
        T,
        CFL,
        IBEG,
        IEND,
        eps,
        GAMMA,
        GAMAM1,
        GMBGMM1,
        GMM1BGM,
        OBGMM1,
        si,
        phi,
        AP,
        Reconstruct,
        boundary_reflective,
        viscosity,
    )
end

function pressure(Q, config)
    RealT = eltype(config.si)
    (; GAMMA) = config
    return RealT(Q.DN^GAMMA)
end

function H_eval(x, config)
    (; GMBGMM1, GMM1BGM, GMBGMM1, phi) = config
    GMBGMM1 * log(GMM1BGM * (GMBGMM1 - phi(x)))
end

function rhoeq(x, config)
    (; GMM1BGM, OBGMM1, phi) = config
    RealT = eltype(config.si)
    return RealT((RealT(1.0) - GMM1BGM * phi(x))^OBGMM1)
end

function rhoeqpert(x, config)
    (; si) = config
    RealT = eltype(si)
    return RealT((rhoeq(x, config) + si * exp(RealT(-100.0) * (x - RealT(0.5))^2)))
end

function rho_hydro(rhoi::RealT, xi::RealT, x::RealT, config) where {RealT<:Real}# xi is the cell center, x is the face
    (; GAMAM1, GMM1BGM, OBGMM1, phi) = config
    return RealT((rhoi^GAMAM1 + GMM1BGM * (phi(xi) - phi(x)))^OBGMM1)
end

function p_hydro(rhoi::RealT, xi::RealT, config) where {RealT<:Real}# rhoi: density at xi, xi: cell center 
    (; GAMMA) = config
    return RealT(rhoi^GAMMA * exp(-H_eval(xi, config)))
end

function p_hydro_inv(phydro::RealT, xi::RealT, config) where {RealT<:Real} # rhoi: density at xi, xi: cell center 
    (; GAMMA) = config
    return RealT((phydro * exp(H_eval(xi, config)))^(1 / GAMMA))
end

function rho_wb(rhoi::RealT, xi::RealT, config) where {RealT<:Real}
    (; phi) = config
    return RealT(rhoi * phi(xi) * exp(-H_eval(xi, config)))
end

function rho_wb_inv(rhoi::RealT, giph::RealT, config) where {RealT<:Real}
    (; phi) = config
    return RealT(rhoi * exp(H_eval(giph, config)) / phi(xi))
end

function new_delt(Q::Vector{DATA1D}, G::Vector{Grid1D}, config::SolverConfig1D)
    eig_max = -1.0e10
    RealT = eltype(config.si)
    for i = config.IBEG:config.IEND
        lambdax = abs(2.0 * Q[i].VX / Q[i].DN)
        eig = lambdax / config.dx

        eig_max = max(eig_max, eig)
    end
    return RealT(config.CFL / eig_max)
end

function initialize_grid(config::SolverConfig1D)
    G = Vector{Grid1D}(undef, config.XNUM_TOT)
    RealT = eltype(config.si)
    for i = 1:config.XNUM_TOT
        xi = RealT(config.XMIN + (i - config.NG - RealT(0.5)) * config.dx)
        G[i] = Grid1D(xi)
    end
    return G
end

function oned_well_balance!(Q, G::Vector{Grid1D}, config::SolverConfig1D)
    RealT = eltype(config.si)
    for i = config.IBEG:config.IEND
        dens = rhoeq(G[i].x, config)
        Q[i] = DATA1D(dens, RealT(0))
    end
end

function one_pert_hydro!(Q::Vector{DATA1D}, G::Vector{Grid1D}, config::SolverConfig1D)
    RealT = eltype(config.si)
    for i = config.IBEG:config.IEND
        dens = rhoeqpert(G[i].x, config)
        Q[i] = DATA1D(dens, RealT(0))
    end
end

function sod!(Q::Vector{DATA1D}, G::Vector{Grid1D}, config::SolverConfig1D)
    (; si) = config
    RealT = eltype(si)
    for i = config.IBEG:config.IEND
        if G[i].x <= RealT(0.25)
            dens = RealT(1.0)
        elseif G[i].x >= RealT(0.75)
            dens = RealT(1.0)
        else
            dens = RealT(1.0) + si
        end
        mom = RealT(1.0)
        Q[i] = DATA1D(dens, mom)
    end
end

function boundary_calculation!(Q::Vector{DATA1D}, G::Vector{Grid1D}, config::SolverConfig1D)
    # Left boundary
    (; boundary_reflective) = config
    j = 1
    for i = (config.NG):-1:1
        den = Q[i+j].DN
        if boundary_reflective == true
            vel = -Q[i+j].VX
        else
            vel = Q[i+j].VX
        end
        Q[i] = DATA1D(den, vel)
        j = j + 2
    end
    # Right boundary
    j = 1
    for i = (config.IEND+1):config.XNUM_TOT
        den = Q[i-j].DN
        if boundary_reflective == true
            vel = -Q[i-j].VX
        else
            vel = Q[i-j].VX
        end
        Q[i] = DATA1D(den, vel)
        j = j + 2
    end
end

function fluxt(Q::DATA1D, config::SolverConfig1D)
    RealT = eltype(config.si)
    P = pressure(Q, config) / config.eps2
    if config.AP
        den = RealT(0.0)
    else
        den = Q.VX
    end
    vel = Q.VX^2 / Q.DN + P
    return DATA1D(den, vel)
end

function flux(Q::DATA1D, config::SolverConfig1D)
    return DATA1D(Q.VX, Q.DN / config.eps2)
end

function max_speedx(Q::DATA1D, config)
    RealT = eltype(config.si)
    (; GAMMA, GAMAM1, eps) = config
    if config.AP
        return RealT(2.0) * abs(Q.VX / Q.DN)
    else
        a = sqrt(GAMMA * Q.DN^(GAMAM1)) / eps
        return (max(Q.VX / Q.DN + a, Q.VX / Q.DN - a))
    end
end

function comp_phi_cc!(
    PHI::Vector{RealT},
    G::Vector{Grid1D},
    config::SolverConfig1D,
) where {RealT<:Real}
    (; phi) = config
    for i = config.IBEG:config.IEND
        PHI[i] = phi(G[i].x)
    end
    # boundary conditions   # same as density
    PHI[config.IBEG-1] = PHI[config.IBEG]
    PHI[config.IBEG-2] = PHI[config.IBEG+1]

    PHI[config.IEND+1] = PHI[config.IEND]
    PHI[config.IEND+2] = PHI[config.IEND-1]
end

function comp_h_cc!(
    H::Vector{RealT},
    G::Vector{Grid1D},
    config::SolverConfig1D,
) where {RealT<:Real}
    (; phi, GMBGMM1, GMM1BGM) = config
    for i = config.IBEG:config.IEND
        H[i] = GMBGMM1 * log(GMM1BGM * (GMBGMM1 - phi(G[i].x)))
    end
    # boundary conditions   # same as density
    H[config.IBEG-1] = H[config.IBEG]
    H[config.IBEG-2] = H[config.IBEG+1]

    H[config.IEND+1] = H[config.IEND]
    H[config.IEND+2] = H[config.IEND-1]
end

function source_calculation!(
    Q,
    G::Vector{Grid1D},
    PHI::Vector{RealT},
    H::Vector{RealT},
    ES::Vector{RealT},
    LS::Vector{RealT},
    config::SolverConfig1D,
    WB::Int,
) where {RealT<:Real}
    (; phi) = config
    for i = config.IBEG:config.IEND

        if WB == 1
            giph = 0.5 * (G[i+1].x + G[i].x)
            gimh = 0.5 * (G[i].x + G[i-1].x)

            exphip1 = exp(H_eval(giph, config))
            exphi = exp(H[i])
            exphim1 = exp(H_eval(gimh, config))


            ES[i] = pressure(Q[i], config) * (exphip1 - exphim1) / (exphi * config.eps2)
        else
            if i == config.IBEG
                ES[i] = -Q[i].DN * (phi(G[i+1].x) - phi(G[i].x)) / config.eps2
            elseif i == config.IEND
                ES[i] = -Q[i].DN * (phi(G[i].x) - phi(G[i-1].x)) / config.eps2
            else
                ES[i] = -0.5 * Q[i].DN * (phi(G[i+1].x) - phi(G[i-1].x)) / config.eps2
            end
        end
        # Linear source (for elliptic correction)

        if i == config.IBEG
            rhoeqiph =
                RealT(0.5) * (rhoeq(G[i+1].x, config) + rhoeq(G[i].x, config)) / config.eps2
            rhoeqimh =
                RealT(0.5) * (rhoeq(G[i].x, config) + rhoeq(G[i].x, config)) / config.eps2
        elseif i == config.IEND
            rhoeqiph =
                RealT(0.5) * (rhoeq(G[i].x, config) + rhoeq(G[i].x, config)) / config.eps2
            rhoeqimh =
                RealT(0.5) * (rhoeq(G[i].x, config) + rhoeq(G[i-1].x, config)) / config.eps2
        else
            rhoeqiph =
                RealT(0.5) * (rhoeq(G[i+1].x, config) + rhoeq(G[i].x, config)) / config.eps2
            rhoeqimh =
                RealT(0.5) * (rhoeq(G[i].x, config) + rhoeq(G[i-1].x, config)) / config.eps2
        end

        rhoBrhoeq = Q[i].DN / rhoeq(G[i].x, config)
        LS[i] = rhoBrhoeq * (rhoeqiph - rhoeqimh)
    end

end

function linear_recovery!(Q::Vector{DATA1D}, Qx::Vector{DATA1D}, config::SolverConfig1D)

    RealT = eltype(config.si)
    if config.Reconstruct  # normal recovery, just central differences
        for i = 2:config.XNUM_TOT-1
            Qx[i] = DATA1D(
                RealT(0.5) * (Q[i+1].DN - Q[i-1].DN) / config.dx,
                RealT(0.5) * (Q[i+1].VX - Q[i-1].VX) / config.dx,
            )
        end
    else    # null recovery, reduces to 1st order scheme
        for i = config.IBEG:config.IEND
            Qx[i] = DATA1D(RealT(0.0), RealT(0.0))
        end
    end
end

function flux_calculation!(
    Q::Vector{DATA1D},
    flxt::Vector{DATA1D},
    flx::Vector{DATA1D},
    H::Vector{RealT},
    G::Vector{Grid1D},
    config::SolverConfig1D,
    dt::RealT,
    WB::Int,
) where {RealT<:Real}
    (; GAMMA) = config
    # Initialize fluxes
    fill!(flxt, DATA1D(RealT))
    fill!(flx, DATA1D(RealT))

    Qhydro = Vector{DATA1D}(undef, config.XNUM_TOT)
    Qx = Vector{DATA1D}(undef, config.XNUM_TOT)
    fill!(Qhydro, DATA1D(RealT))
    fill!(Qx, DATA1D(RealT))

    if WB == 1
        for i = config.IBEG:config.IEND
            Qhydro[i] = DATA1D(p_hydro(Q[i].DN, G[i].x, config), Q[i].VX) # here density is replaced by pressure * exp(-H)
        end
        Qhydro[config.IBEG-1] =
            DATA1D(Q[config.IBEG-1].DN^GAMMA * exp(-H[config.IBEG-1]), Q[config.IBEG-1].VX)
        Qhydro[config.IBEG-2] =
            DATA1D(Q[config.IBEG-2].DN^GAMMA * exp(-H[config.IBEG-2]), Q[config.IBEG-2].VX)


        Qhydro[config.IEND+1] =
            DATA1D(Q[config.IEND+1].DN^GAMMA * exp(-H[config.IEND+1]), Q[config.IEND+1].VX)
        Qhydro[config.IEND+2] =
            DATA1D(Q[config.IEND+2].DN^GAMMA * exp(-H[config.IEND+2]), Q[config.IEND+2].VX)
    else
        Qhydro .= Q
    end

    # reconstruction slopes
    linear_recovery!(Qhydro, Qx, config)

    for i = config.IBEG-1:config.IEND
        # Reconstruction
        Qphmhydro = DATA1D(
            Qhydro[i].DN + RealT(0.5) * Qx[i].DN * config.dx,
            Qhydro[i].VX + RealT(0.5) * Qx[i].VX * config.dx,
        )

        Qphphydro = DATA1D(
            Qhydro[i+1].DN - RealT(0.5) * Qx[i+1].DN * config.dx,
            Qhydro[i+1].VX - RealT(0.5) * Qx[i+1].VX * config.dx,
        )

        if WB == 1
            # Back to primtive variables
            giph = 0.5 * (G[i].x + G[i+1].x)

            Qphm = DATA1D(p_hydro_inv(Qphmhydro.DN, giph, config), Qphmhydro.VX)
            Qphp = DATA1D(p_hydro_inv(Qphphydro.DN, giph, config), Qphphydro.VX)
        else
            Qphm = DATA1D(Qphmhydro.DN, Qphmhydro.VX)
            Qphp = DATA1D(Qphphydro.DN, Qphphydro.VX)
        end

        # Wave speeds
        a_lft = max_speedx(Qphm, config)
        a_rgt = max_speedx(Qphp, config)
        ax = max(a_lft, a_rgt)
        #ax = RealT(0.0)   # no diffusion 

        # Compute Explicit fluxes
        fluxt_lft = fluxt(Qphm, config)
        fluxt_rgt = fluxt(Qphp, config)

        # F_{i+1/2}

        if config.AP
            fluxt_face = DATA1D(
                RealT(0.0),
                RealT(0.5) * (fluxt_rgt.VX + fluxt_lft.VX - ax * (Qphp.VX - Qphm.VX)),
            )
        else
            fluxt_face = DATA1D(
                RealT(0.5) * (fluxt_rgt.DN + fluxt_lft.DN - ax * (Qphp.DN - Qphm.DN)),
                RealT(0.5) * (fluxt_rgt.VX + fluxt_lft.VX - ax * (Qphp.VX - Qphm.VX)),
            )
        end

        # Update fluxes at cell centers
        flxt[i] = DATA1D(flxt[i].DN + fluxt_face.DN, flxt[i].VX + fluxt_face.VX)

        # Compute Implicit Flux
        if config.AP
            flux_lft = flux(Q[i], config)
            flux_rgt = flux(Q[i+1], config)

            flux_face = DATA1D(
                RealT(0.5) * (flux_rgt.DN + flux_lft.DN),
                RealT(0.5) * (flux_rgt.VX + flux_lft.VX),
            )

            # Update fluxes at cell centers
            flx[i] = DATA1D(flx[i].DN + flux_face.DN, flx[i].VX + flux_face.VX)
        else
            flux_face = DATA1D(0.0, 0.0)
            flx[i] = DATA1D(0.0, 0.0)
        end

        if i < config.IEND
            flxt[i+1] = DATA1D(flxt[i+1].DN - fluxt_face.DN, flxt[i+1].VX - fluxt_face.VX)
            flx[i+1] = DATA1D(flx[i+1].DN - flux_face.DN, flx[i+1].VX - flux_face.VX)
        end
    end
end

function rk_update!(
    Q0::Vector{DATA1D},
    Qh::Vector{DATA1D},
    flxt::Vector{DATA1D},
    flx::Vector{DATA1D},
    ES::Vector{RealT},
    LS::Vector{RealT},
    config::SolverConfig1D,
    dt::RealT,
    c1::RealT,
    c2::RealT,
    G::Vector{Grid1D},
) where {RealT<:Real}

    DELTAT1 = dt * c1
    if config.AP
        DELTAT2 = dt * c2
    else
        DELTAT2 = RealT(0.0)
    end

    for i = config.IBEG:config.IEND
        den =
            Q0[i].DN - DELTAT1 * (flxt[i].DN / config.dx) -
            DELTAT2 * (flx[i].DN / config.dx)

        vel =
            Q0[i].VX - DELTAT1 * (flxt[i].VX - ES[i]) / config.dx +
            DELTAT2 * (flx[i].VX - LS[i]) / config.dx
        Qh[i] = DATA1D(den, vel)
    end
end

function thomas_solver_1D!(
    a::Vector{RealT},
    b::Vector{RealT},
    c::Vector{RealT},
    d::Vector{RealT},
    sol::Vector{RealT},
    XNUM::Int,
) where {RealT<:Real}

    cp = Vector{RealT}(undef, XNUM)
    dp = Vector{RealT}(undef, XNUM)

    cp[1] = c[1] / b[1]
    dp[1] = d[1] / b[1]

    for i = 2:XNUM
        denom = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    end

    sol[XNUM] = dp[XNUM]
    for i = (XNUM-1):-1:1
        sol[i] = dp[i] - cp[i] * sol[i+1]
    end
end

function elliptic_solver!(
    Qh::Vector{DATA1D},
    G::Vector{Grid1D},
    config::SolverConfig1D,
    DELTAT::RealT,
    an::RealT,
) where {RealT<:Real}

    cl = Vector{RealT}(undef, config.XNUM)
    cm = Vector{RealT}(undef, config.XNUM)
    cr = Vector{RealT}(undef, config.XNUM)
    rhs = Vector{RealT}(undef, config.XNUM)
    sol = Vector{RealT}(undef, config.XNUM)

    cx = DELTAT * DELTAT * an * an / (config.dx * config.dx * config.eps2)
    c0x = RealT(0.5) * an * DELTAT / config.dx


    # Left boundary (i = 0 in C, i = 1 in our 1-based interior indexing)
    i = 1
    ig = i + config.NG  # Ghost index

    rhoeqavgiph = rhoeq(G[ig+1].x, config) + rhoeq(G[ig].x, config)
    rhoeqavgimh = rhoeq(G[ig].x, config) + RealT(0.0)

    drhroeqiph = rhoeq(G[ig+1].x, config) - rhoeq(G[ig].x, config)
    drhroeqimh = rhoeq(G[ig].x, config) - RealT(0.0)

    cl[i] = RealT(0.0)
    cm[i] = RealT(1.0) + cx * (RealT(1.0) + drhroeqiph / rhoeqavgiph)
    cr[i] = -cx * (RealT(1.0) - drhroeqiph / rhoeqavgiph)
    rhs[i] = Qh[ig].DN - c0x * (Qh[ig+1].VX - Qh[ig-1].VX)
    # Interior points
    for i = 2:config.XNUM-1
        ig = i + config.NG

        rhoeqavgiph = rhoeq(G[ig+1].x, config) + rhoeq(G[ig].x, config)
        rhoeqavgimh = rhoeq(G[ig].x, config) + rhoeq(G[ig-1].x, config)

        drhroeqiph = rhoeq(G[ig+1].x, config) - rhoeq(G[ig].x, config)
        drhroeqimh = rhoeq(G[ig].x, config) - rhoeq(G[ig-1].x, config)

        cl[i] = -cx - cx * drhroeqimh / rhoeqavgimh
        cm[i] =
            RealT(1.0) + RealT(2.0) * cx + cx * drhroeqiph / rhoeqavgiph -
            cx * drhroeqimh / rhoeqavgimh
        cr[i] = -cx + cx * drhroeqiph / rhoeqavgiph
        rhs[i] = Qh[ig].DN - c0x * (Qh[ig+1].VX - Qh[ig-1].VX)
    end

    # Right boundary
    i = config.XNUM
    ig = i + config.NG

    rhoeqavgiph = RealT(0.0) + rhoeq(G[ig].x, config)
    rhoeqavgimh = rhoeq(G[ig].x, config) + rhoeq(G[ig-1].x, config)

    drhroeqiph = RealT(0.0) - rhoeq(G[ig].x, config)
    drhroeqimh = rhoeq(G[ig].x, config) - rhoeq(G[ig-1].x, config)

    cl[i] = -cx * (RealT(1.0) + drhroeqimh / rhoeqavgimh)
    cm[i] = RealT(1.0) + cx * (RealT(1.0) - drhroeqimh / rhoeqavgimh)
    cr[i] = RealT(0.0)
    rhs[i] = Qh[ig].DN - c0x * (Qh[ig+1].VX - Qh[ig-1].VX)


    # Solve tridiagonal system
    thomas_solver_1D!(cl, cm, cr, rhs, sol, config.XNUM)

    # Update solution
    for i = 1:config.XNUM
        ig = i + config.NG
        Qh[ig] = DATA1D(sol[i], Qh[ig].VX)
    end
end

function velocity_update!(
    Q::Vector{DATA1D},
    G::Vector{Grid1D},
    LS::Vector{RealT},
    config::SolverConfig1D,
    dt::RealT,
    an::RealT,
) where {RealT<:Real}

    cx = dt * an / config.dx

    for i = config.IBEG:config.IEND
        rhoiph = RealT(0.5) * (Q[i+1].DN + Q[i].DN)
        rhoimh = RealT(0.5) * (Q[i].DN + Q[i-1].DN)

        diff = RealT(((rhoiph - rhoimh) / config.eps2 - LS[i]))
        Q[i] = DATA1D(Q[i].DN, Q[i].VX - cx * diff)
    end
end

function add_viscosity!(
    Q::Vector{DATA1D},
    Qnm1::Vector{DATA1D},
    G::Vector{Grid1D},
    config::SolverConfig1D,
    dt::RealT,
    an::RealT,
) where {RealT<:Real}

    (; GAMAM1, eps, viscosity) = config
    eigx_max = RealT(0.0)
    for i = config.IBEG:config.IEND

        lambdaxp = Qnm1[i].VX / Qnm1[i].DN + sqrt(Qnm1[i].DN^(GAMAM1)) / eps
        lambdaxm = Qnm1[i].VX / Qnm1[i].DN - sqrt(Qnm1[i].DN^(GAMAM1)) / eps
        lambdax = max(lambdaxp, lambdaxm)

        eigx = lambdax
        eigx_max = max(eigx, eigx_max)
    end

    ax = bx = config.dx / dt

    for i = config.IBEG:config.IEND
        rhoiph = 0.5 * (Qnm1[i+1].VX + Qnm1[i].VX - ax * (Qnm1[i+1].DN - Qnm1[i].DN))
        rhoimh = 0.5 * (Qnm1[i].VX + Qnm1[i-1].VX - bx * (Qnm1[i].DN - Qnm1[i-1].DN))
        if viscosity
            den = Q[i].DN - an * dt * (rhoiph - rhoimh) / config.dx
        else
            den = Q[i].DN
        end
        Q[i] = DATA1D(den, Q[i].VX)
    end
end

function setup_tableau(IntegratorType::Int, RealT; gm = 0.5)
    if IntegratorType == 1
        #DIRK(1,1,1)
        stages = 2
        at = RealT.([
            0.0 0.0
            1.0 0.0
        ])
        a = RealT.([
            0.0 0.0
            0.0 1.0
        ])

        omegat = RealT.([1.0, 0.0])
        omega = RealT.([0.0, 1.0])
        println("IMEX-RK tableau: = DIRK(1,1,1)")
    elseif IntegratorType == 2
        #DP1A1
        gm = RealT(0.8)
        stages = 2
        at = RealT.([
            0.0 0.0
            1.0 0.0
        ])
        a = RealT.([
            gm 0.0
            1.0-gm gm
        ])

        omegat = RealT.([1.0, 0.0])
        omega = RealT.([1.0 - gm, gm])
        println("IMEX-RK tableau: = DP1A1")
    elseif IntegratorType == 3
        # DP2A1
        stages = 4
        at = RealT.([
            0.0 0.0 0.0 0.0
            1.0/3.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0
            0.5 0.0 0.5 0.0
        ])
        a =
            RealT.(
                [
                    0.5 0.0 0.0 0.0
                    1.0/6.0 0.5 0.0 0.0
                    -0.5 0.5 0.5 0.0
                    3.0/2.0 -3.0/2.0 0.5 0.5
                ]
            )

        omegat = RealT.([0.5, 0.0, 0.5, 0.0])
        omega = RealT.([3.0 / 2.0, -3.0 / 2.0, 0.5, 0.5])
        println("IMEX-RK tableau: = DP2A1")
    elseif IntegratorType == 4
        # DP2A2
        #gm = 2.0 # monotone
        gm = gm
        stages = 4
        at = RealT.([
            0.0 0.0 0.0 0.0
            0.0 0.0 0.0 0.0
            0.0 1.0 0.0 0.0
            0.0 0.5 0.5 0.0
        ])
        a = RealT.([
            gm 0.0 0.0 0.0
            -gm gm 0.0 0.0
            0.0 1.0-gm gm 0.0
            0.0 0.5 0.5-gm gm
        ])

        omegat = RealT.([0.0, 0.5, 0.5, 0.0])
        omega = RealT.([0.0, 0.5, 0.5 - gm, gm])
        println("IMEX-RK tableau: = DP2A2")
    elseif IntegratorType == 5
        rq = 1.0 - 1.0 / sqrt(2.0)
        d = 1.0 - 1.0 / (2.0 * rq)
        stages = 3
        at = RealT.([
            0.0 0.0 0.0
            rq 0.0 0.0
            d 1.0-d 0.0
        ])

        a = RealT.([
            0.0 0.0 0.0
            0.0 rq 0.0
            0.0 1.0-rq rq
        ])

        omegat = RealT.([d, 1.0 - d, 0.0])
        omega = RealT.([0.0, 1.0 - rq, rq])
        println("IMEX-RK tableau: = ARS(2,2,2)")
    elseif IntegratorType == 6
        gm = 0.5
        dl = gm / (1.0 - gm)
        stages = 3
        at = RealT.([
            0.0 0.0 0.0
            dl 0.0 0.0
            1.0 0.0 0.0
        ])

        a = RealT.([
            0.0 0.0 0.0
            0.0 gm 0.0
            0.0 1.0-gm gm
        ])

        omegat = RealT.([1.0, 0.0, 0.0])
        omega = RealT.([0.0, 1.0 - gm, gm])
        println("IMEX-RK tableau: = DP-ARS(1,2,1)")
    end
    return IMEXTableau(stages, at, a, omegat, omega)
end

function write_data(
    Q,
    G::Vector{Grid1D},
    config::SolverConfig1D,
    filename::String = "output.dat",
)
    open(filename, "w") do fptr
        for i = config.IBEG:config.IEND
            @printf(
                fptr,
                "%.5e\t%.22e\t%.22e\n",
                Float64(G[i].x),
                Float64(Q[i].DN),
                Float64(Q[i].VX)
            )
        end
    end
end

function well_balance_check(
    Q::Vector{DATA1D},
    G::Vector{Grid1D},
    config::SolverConfig1D,
    filename::String = "wb_check.dat",
)
    open(filename, "w") do fptr
        for i = config.IBEG:config.IEND
            ddiff = sqrt((Q[i].DN - rhoeq(G[i].x, config))^2)
            velsq = sqrt(Q[i].VX^2)
            @printf(
                fptr,
                "%.5e\t%.22e\t%.22e\n",
                Float64(G[i].x),
                Float64(ddiff),
                Float64(velsq)
            )
        end
    end
end

function main(;
    PROBLEM,
    IntegratorType,
    XNUMB,
    t_final,
    eps,
    GAMMA,
    Well_balanced = 1,
    si,
    phi,
    factor,
    cfl_condition,
    AP,
    Reconstruct,
    boundary_reflective = true,
    CFL = 0.9,
    viscosity = false,
    gm = 0.7,
)
    # Configuration
    # Run Parameters setup 
    # Initial conditions - choose one
    # "PROBLEM" ?  1 = well-balance, 2 = perturbed, 3 = riemann Problem

    # "IntegratorType" ? 1 : DIRK(1,1,1), 2 : DP1A1, 3 : DP2A1 , 4: DP2A2 5 : ARS(2,2,2), 6 : DP-ARS(1,2,1)
    RealT = eltype(si)
    # Configuration
    config = create_config(
        XNUMB,
        t_final,
        eps,
        GAMMA,
        si,
        phi,
        AP,
        Reconstruct,
        boundary_reflective,
        RealT(CFL),
        viscosity,
    )
    println("Configuration created")
    println("XNUM = ", config.XNUM, ", dx = ", config.dx)
    println("eps = ", eps, ", eps2 = ", config.eps2)

    # Initialize grid
    G = initialize_grid(config)
    println("Grid initialized")

    count = 0   # no. of time steps
    time = RealT(0.0)   # initial or current time
    DELTAT = config.dx * factor # Initial time step


    # Allocate arrays
    Q = Vector{DATA1D}(undef, config.XNUM_TOT)
    Q0 = Vector{DATA1D}(undef, config.XNUM_TOT)
    Qx = Vector{DATA1D}(undef, config.XNUM_TOT)
    Qs1 = Vector{DATA1D}(undef, config.XNUM_TOT)
    Qs2 = Vector{DATA1D}(undef, config.XNUM_TOT)
    Qs3 = Vector{DATA1D}(undef, config.XNUM_TOT)
    Qs4 = Vector{DATA1D}(undef, config.XNUM_TOT)
    Qunpe = Vector{DATA1D}(undef, config.XNUM_TOT)
    Qdiff = Vector{DATA1D}(undef, config.XNUM_TOT)
    Qdiff0 = Vector{DATA1D}(undef, config.XNUM_TOT)
    Qdiff0 = Vector{DATA1D}(undef, config.XNUM_TOT)
    Qdiff0 = [DATA1D(zero(RealT), zero(RealT)) for _ = 1:config.XNUM_TOT]
    Qunpe = [DATA1D(zero(RealT), zero(RealT)) for _ = 1:config.XNUM_TOT]
    flxt = Vector{DATA1D}(undef, config.XNUM_TOT)
    flx = Vector{DATA1D}(undef, config.XNUM_TOT)

    ES = Vector{RealT}(undef, config.XNUM_TOT)
    LS = Vector{RealT}(undef, config.XNUM_TOT)

    PHI = Vector{RealT}(undef, config.XNUM_TOT)
    H = Vector{RealT}(undef, config.XNUM_TOT)

    println("Arrays allocated")

    if PROBLEM == 1
        oned_well_balance!(Q0, G, config)
        println("Well-balanced initial condition set")

        #well_balance_check(Q0, G, config, "wb_check_0.dat")
        ##println("Well-balance check written to wb_check_0.dat")
    elseif PROBLEM == 2
        oned_well_balance!(Qunpe, G, config)
        one_pert_hydro!(Q0, G, config)
        println("Perturbed hydrostatic initial condition set")

        for i = config.IBEG:config.IEND
            den_diff = abs(Qunpe[i].DN - Q0[i].DN)
            Qdiff0[i] = DATA1D(den_diff, RealT(0))
        end
        #   write_data(Qdiff0, G, config, "dendiff_0.dat")
        #   println("\n Initial perturbation written to dendiff_0.dat")
    else
        sod!(Q0, G, config)
        println("Sod shock tube initial condition set")
    end

    # Apply boundary conditions
    boundary_calculation!(Q0, G, config)

    # Write initial data output
    #write_data(Q0, G, config, "output_0.dat")
    #println("\n Initial data written to output_0.dat")

    if PROBLEM == 1
        # Open file for writing time evolve of balance
        name = @sprintf("time_balance.dat")
        fptr = open(name, "w")
        # Initialize accumulators
        l2_dn = 0.0
        l2_v = 0.0
        for i = config.IBEG:config.IEND
            l2_dn += (Q0[i].DN - rhoeq(G[i].x, config))^2
            l2_v += Q0[i].VX^2
        end
        # Compute RMS values
        l2_dn = sqrt(l2_dn / config.XNUM)
        l2_v = sqrt(l2_v / config.XNUM)

        @show (l2_dn)
        @show (l2_v)
        # Write to file
        @printf(fptr, "%.5e\t%.24e\t%.24e\n", time, l2_dn, l2_v)
    end
    ##########

    tableau = setup_tableau(IntegratorType, RealT, gm = gm)
    println("\nStarting time integration...")

    # Copy current state to Q0
    Q .= Q0

    # Pre-compute gravitational potential and enthalpy at cell centers
    comp_phi_cc!(PHI, G, config)
    comp_h_cc!(H, G, config)

    while time < config.T
        count += 1

        if (time + DELTAT) > config.T
            DELTAT = config.T - time
        end

        #@printf(
        #    "\nStep %d: DELTAT = %.6e, time = %.6e\n",
         #   count,
         #   Float64(DELTAT),
         #   Float64(time)
       # )

        # Zero out fluxes
        fill!(flxt, DATA1D(RealT))
        fill!(flx, DATA1D(RealT))
        fill!(Qx, DATA1D(RealT))
        fill!(ES, RealT(0.0))
        fill!(LS, RealT(0.0))

        # No flux calculation needed for first stage
        # Update explicit part of stage-1
        rk_update!(Q, Qs1, flxt, flx, ES, LS, config, DELTAT, RealT(0.0), RealT(0.0), G)
        boundary_calculation!(Qs1, G, config)

        # Solve elliptic problem
        elliptic_solver!(Qs1, G, config, DELTAT, tableau.a[1, 1])
        boundary_calculation!(Qs1, G, config)

        # Compute source terms
        source_calculation!(Qs1, G, PHI, H, ES, LS, config, Well_balanced)

        if config.AP
            # Update momentum
            velocity_update!(Qs1, G, LS, config, DELTAT, tableau.a[1, 1])
            add_viscosity!(Qs1, Q, G, config, DELTAT, tableau.a[1, 1])
            boundary_calculation!(Qs1, G, config)
        end

        # Flux calculation with stage-1 data
        flux_calculation!(Qs1, flxt, flx, H, G, config, DELTAT, Well_balanced)

        # Update explicit part of stage-2
        rk_update!(
            Q,
            Qs2,
            flxt,
            flx,
            ES,
            LS,
            config,
            DELTAT,
            tableau.at[2, 1],
            tableau.a[2, 1],
            G,
        )
        boundary_calculation!(Qs2, G, config)

        if tableau.stages >= 3
            rk_update!(
                Q,
                Qs3,
                flxt,
                flx,
                ES,
                LS,
                config,
                DELTAT,
                tableau.at[3, 1],
                tableau.a[3, 1],
                G,
            )
            boundary_calculation!(Qs3, G, config)
        end

        if tableau.stages == 4
            rk_update!(
                Q,
                Qs4,
                flxt,
                flx,
                ES,
                LS,
                config,
                DELTAT,
                tableau.at[4, 1],
                tableau.a[4, 1],
                G,
            )
            boundary_calculation!(Qs4, G, config)
        end

        if config.AP
            # Solve elliptic problem
            elliptic_solver!(Qs2, G, config, DELTAT, tableau.a[2, 2])
            boundary_calculation!(Qs2, G, config)
        end
        #Compute source terms
        source_calculation!(Qs2, G, PHI, H, ES, LS, config, Well_balanced)

        if config.AP
            # Update momentum
            velocity_update!(Qs2, G, LS, config, DELTAT, tableau.a[2, 2])
            add_viscosity!(Qs2, Qs1, G, config, DELTAT, tableau.a[2, 2])
            boundary_calculation!(Qs2, G, config)
        end

        if tableau.stages == 2
            Q .= Qs2
        end

        if tableau.stages >= 3
            # Update explicit part of stage-3
            # Flux calculation with stage-2 data
            flux_calculation!(Qs2, flxt, flx, H, G, config, DELTAT, Well_balanced)    # reconstruction to be added later

            rk_update!(
                Qs3,
                Qs3,
                flxt,
                flx,
                ES,
                LS,
                config,
                DELTAT,
                tableau.at[3, 2],
                tableau.a[3, 2],
                G,
            )
            boundary_calculation!(Qs3, G, config)

            if tableau.stages == 4
                rk_update!(
                    Qs4,
                    Qs4,
                    flxt,
                    flx,
                    ES,
                    LS,
                    config,
                    DELTAT,
                    tableau.at[4, 2],
                    tableau.a[4, 2],
                    G,
                )
                boundary_calculation!(Qs4, G, config)
            end

            if config.AP
                # Solve elliptic problem
                elliptic_solver!(Qs3, G, config, DELTAT, tableau.a[3, 3])
                boundary_calculation!(Qs3, G, config)
            end
            #Compute source terms
            source_calculation!(Qs3, G, PHI, H, ES, LS, config, Well_balanced)

            if config.AP
                # Update momentum
                velocity_update!(Qs3, G, LS, config, DELTAT, tableau.a[3, 3])
                add_viscosity!(Qs3, Qs2, G, config, DELTAT, tableau.a[3, 3])
                boundary_calculation!(Qs3, G, config)
            end

        end
        if tableau.stages == 3
            Q .= Qs3
        end

        if tableau.stages == 4
            # Flux calculation with stage-3 data
            flux_calculation!(Qs3, flxt, flx, H, G, config, DELTAT, Well_balanced)    # reconstruction to be added later

            rk_update!(
                Qs4,
                Qs4,
                flxt,
                flx,
                ES,
                LS,
                config,
                DELTAT,
                tableau.at[4, 3],
                tableau.a[4, 3],
                G,
            )
            boundary_calculation!(Qs4, G, config)

            if config.AP
                # Solve elliptic problem
                elliptic_solver!(Qs4, G, config, DELTAT, tableau.a[4, 4])
                boundary_calculation!(Qs4, G, config)
            end

            #Compute source terms
            source_calculation!(Qs4, G, PHI, H, ES, LS, config, Well_balanced)

            if config.AP
                # Update momentum
                velocity_update!(Qs4, G, LS, config, DELTAT, tableau.a[4, 4])
                add_viscosity!(Qs4, Qs3, G, config, DELTAT, tableau.a[4, 4])
                boundary_calculation!(Qs4, G, config)
            end


            Q .= Qs4
        end

        if PROBLEM == 1
            l2_dn = RealT(0.0)
            l2_v = RealT(0.0)
            for i = config.IBEG:config.IEND
                l2_dn += (Q[i].DN - rhoeq(G[i].x, config))^2
                l2_v += Q[i].VX^2
            end
            # Compute RMS values
            l2_dn = sqrt(l2_dn / config.XNUM)
            l2_v = sqrt(l2_v / config.XNUM)

            # Write to file
            @printf(fptr, "%.5e\t%.24e\t%.24e\n", time, l2_dn, l2_v)
        end
        time += DELTAT
        if cfl_condition
            DELTAT = new_delt(Q, G, config)
        end
    end

    # Write output
    #write_data(Q, G, config, "output_Q.dat")
    #println("\nOutput written to output_Q.dat")

    if PROBLEM == 1
        #  well_balance_check(Q, G, config, "wb_check.dat")
        # println("Well-balance check written to wb_check.dat")

        @show l2_dn
        @show l2_v
    elseif PROBLEM == 2
        l2_ddiff = RealT(0.0)
        l2_vdiff = RealT(0.0)
        for i = config.IBEG:config.IEND
            den_diff = abs(Qunpe[i].DN - Q[i].DN)
            Qdiff[i] = DATA1D(den_diff, RealT(0))
            l2_ddiff += den_diff^2
            l2_vdiff += Q[i].VX^2
        end

        # Compute RMS values
        l2_dn = sqrt(l2_ddiff / config.XNUM)
        l2_v = sqrt(l2_vdiff / config.XNUM)
        @show l2_dn
        @show l2_v

        #write_data(Qdiff, G, config, "results/dendiff_$(Well_balanced).dat")
        #println("\nOutput written to dendiff.dat")
    elseif PROBLEM == 3
        l2_dn = 0
        l2_v = 0
    end

    return (; l2_dn, l2_v, Q, G, Q0, Qunpe, Qdiff0)
end

function phi_linear(x::RealT) where {RealT<:Real}
    return RealT(x)
end
function phi_quadratic(x::RealT) where {RealT<:Real}
    return RealT(x^2)
end
function phi_sin(x::RealT) where {RealT<:Real}
    return RealT(sin(2.0 * pi * x))
end

function reproduce_well_balanced()
    RealT = Double64
    epsilon_v = (1, 1e-1, 1e-2, 1e-3, 1e-4)
    phi_v = (phi_linear, phi_quadratic, phi_sin)
    phi_names = ("phi_linear", "phi_quadratic", "phi_sin")

    results = [[Tuple{RealT,RealT}((0, 0)) for _ in epsilon_v] for _ in phi_v]
    for (j, phi) in enumerate(phi_v)
        for (i, eps) in enumerate(epsilon_v)
            sol = main(;
                PROBLEM = 1,
                IntegratorType = 4,
                XNUMB = 100,
                t_final = RealT(10),
                eps = RealT(eps),
                GAMMA = RealT(1.4),
                Well_balanced = 1,
                si = RealT(0),
                phi = phi,
                factor = RealT(0.5),
                cfl_condition = false,
                AP = true,
                Reconstruct = true,
            )
            results[j][i] = (sol.l2_dn, sol.l2_v)
        end
    end
    eps_labels =
        (raw"$1.0$", raw"$10^{-1}$", raw"$10^{-2}$", raw"$10^{-3}$", raw"$10^{-4}$")

    for (i, elabel) in enumerate(eps_labels)
        row = elabel
        for j = 1:length(phi_v)
            dn, v = results[j][i]
            row *= @sprintf(" & %.3E & %.3E", dn, v)
        end
        row *= raw" \\"
        println(row)
        println(raw"\hline")
    end
end

function plot_single_comparison(ax; si, epsilon, nwb = true)

    RealT = Double64
    t_final = RealT(0.25)
    si = RealT(si)
    epsilon = RealT(epsilon)
    sol_wb = main(;
        PROBLEM = 2,
        IntegratorType = 4,
        XNUMB = 100,
        t_final = t_final,
        eps = epsilon,
        GAMMA = RealT(1.4),
        Well_balanced = 1,
        si = si,
        phi = phi_linear,
        factor = RealT(0.1),
        cfl_condition = false,
        AP = true,
        Reconstruct = true,
    )
    if nwb
        sol_nwb = main(;
            PROBLEM = 2,
            IntegratorType = 4,
            XNUMB = 100,
            t_final = t_final,
            eps = epsilon,
            GAMMA = RealT(1.4),
            Well_balanced = 0,
            si = si,
            phi = phi_linear,
            factor = RealT(0.1),
            cfl_condition = false,
            AP = true,
            Reconstruct = false,
        )
    else
        sol_nwb = sol_wb
    end

    x_coords = [g.x for g in sol_wb.G]
    rho_wb_plot = [g.DN for g in sol_wb.Q]
    rho_nwb_plot = [g.DN for g in sol_nwb.Q]
    rho_wb_unpe = [g.DN for g in sol_wb.Qunpe]
    rho_nwb_unpe = [g.DN for g in sol_nwb.Qunpe]
    rho0 = [g.DN for g in sol_wb.Qdiff0]


    colors = Makie.wong_colors()
    # Plot
    lines!(
        ax,
        x_coords[3:end-2],
        rho0[3:end-2] ./ si,
        color = colors[3],
        label = L"$T = 0.0$",
    )
    lines!(
        ax,
        x_coords[3:end-2],
        abs.(rho_wb_plot[3:end-2] .- rho_wb_unpe[3:end-2]) ./ si,
        color = colors[1],
        label = L"$T = 0.25$ (WB)",
    )
    if nwb
        lines!(
            ax,
            x_coords[3:end-2],
            abs.(rho_nwb_plot[3:end-2] .- rho_nwb_unpe[3:end-2]) ./ si,
            color = colors[2],
            label = L"$T = 0.25$ (NWB)",
        )
    end

end

#"PROBLEM" ?  1 = well-balance, 2 = perturbed, 3 = riemann Problem
#"IntegratorType" ? 1 : DIRK(1,1,1), 2 : DP1A1, 3 : DP2A1 , 4: DP2A2 5 : ARS(2,2,2), 6 : DP-ARS(1,2,1)
function plot_comparison_wb_nwb_1()
    RealT = Double64
    h = Figure(size = (1100, 500))
    labelsize = 22
    ticklabelsize = 24
    titlesize = 25

    epsilon = 1.0
    si = 1e-3
    XNUMB = 100
    kwargs = (
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), nothing),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        titlesize = titlesize,
        xlabel = L"x_1",
    )

    t1 = format_var(epsilon, name = "\\varepsilon")
    t2 = format_var(si, name = "\\zeta")
    ax1 = Axis(
        h[2, 1];
        xlabel = L"x_1",
        ylabel = L"||\rho(t) - \rho(0)||_{L^2}/\zeta",
        title = L"%$t1,\; %$t2",
        kwargs...,
    )
    plot_single_comparison(ax1; si, epsilon)


    si = 1e-5
    t1 = format_var(epsilon, name = "\\varepsilon")
    t2 = format_var(si, name = "\\zeta")
    ax2 = Axis(h[2, 2]; xlabel = L"x_1", title = L"%$t1,\; %$t2", kwargs...)

    plot_single_comparison(ax2; si, epsilon)

    kwargs = (
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), (-0.04, 1.04)),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        titlesize = titlesize,
        xlabel = L"x_1",
    )

    t1 = format_var(epsilon, name = "\\varepsilon")
    t2 = format_var(si, name = "\\zeta")
    ax3 = Axis(h[2, 3]; xlabel = L"x_1", title = L"%$t1,\; %$t2", kwargs...)

    plot_single_comparison(ax3; si, epsilon)

    Legend(
        h[1, 2],
        ax1,
        framevisible = true,
        labelsize = 25,
        orientation = :horizontal,
        nbanks = 1,
    )

    save("results/plot_comparison_1.png", h)
end

function plot_comparison_wb_nwb_2()
    RealT = Double64
    h2 = Figure(size = (1200, 500))
    labelsize = 22
    ticklabelsize = 24
    titlesize = 25

    epsilon = 1e-1
    si = 1e-2
    XNUMB = 100
    kwargs = (
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), nothing),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        titlesize = titlesize,
        xlabel = L"x_1",
    )

    t1 = format_var(epsilon, name = "\\varepsilon")
    t2 = format_var(si, name = "\\zeta")
    ax1 = Axis(
        h2[2, 1];
        xlabel = L"x_1",
        ylabel = L"||\rho(t) - \rho(0)||_{L^2}/\zeta",
        title = L"%$t1,\; %$t2",
        kwargs...,
    )

    plot_single_comparison(ax1; si, epsilon)
    epsilon = 1e-2
    si = 1e-4
    t1 = format_var(epsilon, name = "\\varepsilon")
    t2 = format_var(si, name = "\\zeta")
    ax = Axis(h2[2, 2]; xlabel = L"x_1", title = L"%$t1,\; %$t2", kwargs...)


    plot_single_comparison(ax; si, epsilon)

    kwargs = (
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), (-0.04, 1.04)),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        titlesize = titlesize,
        xlabel = L"x_1",
    )
    epsilon = 1e-3
    si = 1e-6
    t1 = format_var(epsilon, name = "\\varepsilon")
    t2 = format_var(si, name = "\\zeta")
    ax = Axis(h2[2, 3]; xlabel = L"x_1", title = L"%$t1,\; %$t2", kwargs...)

    plot_single_comparison(ax; si, epsilon, nwb = false)

    Legend(
        h2[1, 2],
        ax1,
        framevisible = true,
        labelsize = 25,
        orientation = :horizontal,
        nbanks = 1,
    )

    save("results/plot_comparison_2.png", h2)
    return nothing
end

function format_var(v_raw; name = "\\varepsilon")
    v = Float64(v_raw)
    exp = floor(Int, log10(v))
    mantissa = v / 10.0^exp

    val_str = if exp == 0
        "$v"
    elseif isapprox(mantissa, 1.0)
        "10^{$exp}"
    else
        "$mantissa \\times 10^{$exp}"
    end

    return "$name = $val_str"
end

function compute_eoc()
    XNUMB_v = (20, 40, 80, 160, 320)
    epsilon_v = (1e-4, 1e-5, 1e-6)
    RealT = Double64

    results = Dict()

    for epsilon in epsilon_v
        errors = RealT[]
        for XNUMB in XNUMB_v
            sol = main(;
                PROBLEM = 2,
                IntegratorType = 4,
                XNUMB = XNUMB,
                t_final = RealT(3.0),
                eps = RealT(epsilon),
                GAMMA = RealT(1.4),
                Well_balanced = 1,
                si = RealT(epsilon),
                phi = phi_linear,
                factor = RealT(0.1),
                cfl_condition = false,
                AP = true,
                Reconstruct = true,
            )
            push!(errors, sol.l2_v)
        end
        results[epsilon] = errors
    end

    println("\nN\t\t", join(["ε = $e" for e in epsilon_v], "\t\t"))
    println("-"^80)

    for (i, N) in enumerate(XNUMB_v)
        row = "$N\t"
        for epsilon in epsilon_v
            errs = results[epsilon]
            if i == 1
                row *= @sprintf("%.2e  (---)\t", errs[i])
            else
                aoc = log2(errs[i-1] / errs[i])
                row *= @sprintf("%.2e (%.2f)\t", errs[i], aoc)
            end
        end
        println(row)
    end
end

function plot_riemann_solution()

    RealT = Double64
    epsilon = RealT(0.3)
    t_final = RealT(0.01)
    sol = main(;
        PROBLEM = 3,
        IntegratorType = 4,
        XNUMB = 1000,
        t_final = t_final,
        eps = epsilon,
        GAMMA = RealT(2),
        Well_balanced = 1,
        si = epsilon^2,
        phi = phi_linear,
        factor = RealT(0.1),
        cfl_condition = true,
        AP = true,
        Reconstruct = true,
        boundary_reflective = false,
        CFL = 0.9,
        gm = 0.5,
    )

    XNUMB = 10000
    sol_ref = main(;
        PROBLEM = 3,
        IntegratorType = 1,
        XNUMB = XNUMB,
        t_final = t_final,
        eps = epsilon,
        GAMMA = RealT(2),
        Well_balanced = 1,
        si = epsilon^2,
        phi = phi_linear,
        factor = RealT(0.1),
        cfl_condition = true,
        AP = false,
        Reconstruct = false,
        boundary_reflective = false,
        CFL = 0.1,
    )
    fig = Figure(size = (1200, 500))
    labelsize = 25
    ticklabelsize = 24
    titlesize = 25

    epsilon = 1e-1
    si = 1e-2
    kwargs = (
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), nothing),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        titlesize = titlesize,
        xlabel = L"x_1",
    )

    ax1 = Axis(fig[1, 1]; xlabel = L"x_1", ylabel = L"\rho", kwargs...)
    x_coords = [g.x for g in sol.G]
    rho = [g.DN for g in sol.Q]
    m = [g.VX for g in sol.Q]
    x_coords_ref = [g.x for g in sol_ref.G]
    rho_ref = [g.DN for g in sol_ref.Q]
    m_ref = [g.VX for g in sol_ref.Q]
    lines!(x_coords_ref[3:end-2], rho_ref[3:end-2])
    lines!(x_coords[3:end-2], rho[3:end-2])
    kwargs = (
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), nothing),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        titlesize = titlesize,
        xlabel = L"x_1",
    )

    ax1 = Axis(fig[1, 2]; xlabel = L"x_1", ylabel = L"q_1", kwargs...)
    x_coords = [g.x for g in sol.G]
    rho = [g.DN for g in sol.Q]
    m = [g.VX for g in sol.Q]
    x_coords_ref = [g.x for g in sol_ref.G]
    rho_ref = [g.DN for g in sol_ref.Q]
    m_ref = [g.VX for g in sol_ref.Q]
    lines!(x_coords_ref[3:end-2], m_ref[3:end-2], label = L"reference $$")
    lines!(x_coords[3:end-2], m[3:end-2], label = L"AP-WB $$")

    axislegend(ax1, position = :rb, framevisible = true, labelsize = 25)
    save("results/riemann_solution.png", fig)
end

plot_comparison_wb_nwb_1()
plot_comparison_wb_nwb_2()
reproduce_well_balanced()
compute_eoc()
