using LinearAlgebra
using CairoMakie
using DelimitedFiles
using DoubleFloats
using Printf
using SparseArrays
using Sparspak
using Sparspak.Problem: Problem, insparse!, outsparse, infullrhs!
using Sparspak.SparseSolver: SparseSolver

struct Data{RealT}
    DN::RealT  # Density
    VX::RealT  # x-velocity
    VY::RealT  # y-velocity
end

Data(RealT) = Data(RealT(0.0), RealT(0.0), RealT(0.0))

struct Grid{RealT}
    x::RealT
    y::RealT
end

struct SolverConfig{RealT,GeopotentialT}
    # Domain
    XMIN::RealT
    XMAX::RealT

    YMIN::RealT
    YMAX::RealT

    XNUM::Int
    YNUM::Int

    # Grid
    NG::Int

    YNUM_TOT::Int
    XNUM_TOT::Int

    dx::RealT
    dy::RealT

    # Physical parameters
    # Time integration
    T::RealT
    CFL::RealT

    # Computed bounds
    IBEG::Int
    IEND::Int

    JBEG::Int
    JEND::Int
    eps::RealT
    eps2::RealT
    GAMMA::RealT
    GAMAM1::RealT
    GMBGMM1::RealT
    GMM1BGM::RealT
    OBGMM1::RealT
    si::RealT
    phi::GeopotentialT
    AP::Bool
    HYDRORECON::Bool
    boundary_reflective::Bool
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
    xno::Int,
    yno::Int,
    t_final::RealT,
    eps,
    GAMMA,
    si,
    phi,
    AP,
    HYDRORECON,
    boundary_reflective,
    CFL,
) where {RealT<:Real}
    XMIN = RealT(0.0)
    XMAX = RealT(1.0)

    YMIN = RealT(0.0)
    YMAX = RealT(1.0)

    XNUM = xno
    YNUM = yno

    NG = 2  # Number of ghost cells

    XNUM_TOT = XNUM + 2 * NG
    dx = RealT((XMAX - XMIN) / XNUM)

    YNUM_TOT = YNUM + 2 * NG
    dy = RealT((YMAX - YMIN) / YNUM)

    T = t_final
    CFL = RealT(0.22)

    IBEG = 1 + NG
    IEND = XNUM + NG
    JBEG = 1 + NG
    JEND = YNUM + NG
    eps2 = eps^2
    GAMAM1 = GAMMA - RealT(1.0)
    GMBGMM1 = GAMMA / GAMAM1
    GMM1BGM = GAMAM1 / GAMMA
    OBGMM1 = RealT(1.0) / GAMAM1

    return SolverConfig(
        XMIN,
        XMAX,
        YMIN,
        YMAX,
        XNUM,
        YNUM,
        NG,
        XNUM_TOT,
        YNUM_TOT,
        dx,
        dy,
        T,
        CFL,
        IBEG,
        IEND,
        JBEG,
        JEND,
        eps,
        eps2,
        GAMMA,
        GAMAM1,
        GMBGMM1,
        GMM1BGM,
        OBGMM1,
        si,
        phi,
        AP,
        HYDRORECON,
        boundary_reflective,
    )
end

function initialize_grid(config::SolverConfig)
    G = Matrix{Grid}(undef, config.XNUM_TOT, config.YNUM_TOT)
    RealT = eltype(config.si)
    for j = 1:config.YNUM_TOT
        yj = RealT(config.YMIN + (j - config.NG - RealT(0.5)) * config.dy)
        for i = 1:config.XNUM_TOT
            xi = RealT(config.XMIN + (i - config.NG - RealT(0.5)) * config.dx)
            G[i, j] = Grid(xi, yj)
        end
    end
    return G
end

function pressure(Q::Data, config)
    (; GAMMA) = config
    return Q.DN^GAMMA
end

function kinetic_energy(Q::Matrix{Data}, G::Matrix{Grid}, config::SolverConfig)
    RealT = eltype(config.si)
    KE = RealT(0.0)
    for j = config.JBEG:config.JEND
        for i = config.IBEG:config.IEND
            KE += RealT(0.5) * (Q[i, j].VX^2 + Q[i, j].VY^2) / Q[i, j].DN
        end
    end
    return KE
end

function new_delt(Q::Matrix{Data}, G::Matrix{Grid}, config::SolverConfig)
    eig_max = -1.0e10
    RealT = eltype(config.si)
    for j = config.JBEG:config.JEND
        for i = config.IBEG:config.IEND
            lambdax = abs(2.0 * Q[i, j].VX / Q[i, j].DN)
            lambday = abs(2.0 * Q[i, j].VY / Q[i, j].DN)
            eig = lambdax / config.dx + lambday / config.dy

            eig_max = max(eig_max, eig)
        end
    end
    return RealT(config.CFL / eig_max)
end

function H_eval(x::RealT, y::RealT, config) where {RealT<:Real}
    (; GMBGMM1, GMM1BGM, phi) = config
    GMBGMM1 * log(GMM1BGM * (GMBGMM1 - phi(x, y)))
end

# Equilibrium density profile
function rhoeq(x::RealT, y::RealT, config) where {RealT<:Real}
    (; GMM1BGM, OBGMM1, phi) = config
    return RealT((RealT(1.0) - GMM1BGM * phi(x, y))^OBGMM1)
end

function p_hydro(rhoij::RealT, xi::RealT, yj::RealT, config) where {RealT<:Real}  # rhoij: density at (xi, yj): cell center 
    (; GAMMA) = config
    return RealT(rhoij^GAMMA * exp(-H_eval(xi, yj, config)))
end


function p_hydro_inv(phydro::RealT, xi::RealT, yj::RealT, config) where {RealT<:Real} # rhoi: density at xi, xi: cell center 
    (; GAMMA) = config
    return RealT((phydro * exp(H_eval(xi, yj, config)))^(1 / GAMMA))
end


function rhoeqpert(x::RealT, y::RealT, config) where {RealT<:Real}
    #si = RealT(0.1)  # Perturbation strength
    (; si) = config
    return RealT(
        rhoeq(x, y, config) + si * exp(-100.0 * ((x - RealT(0.3))^2 + (y - RealT(0.3))^2)),
    )
end

function twod_well_balance!(Q::Matrix{Data}, G::Matrix{Grid}, config::SolverConfig)

    for j = config.JBEG:config.JEND
        for i = config.IBEG:config.IEND
            dens = rhoeq(G[i, j].x, G[i, j].y, config)
            Q[i, j] = Data(dens, zero(eltype(dens)), zero(eltype(dens)))
        end
    end
end

function twod_pert_hydro!(Q::Matrix{Data}, G::Matrix{Grid}, config::SolverConfig)#, time::RealT)
    for j = config.JBEG:config.JEND
        for i = config.IBEG:config.IEND
            dens = rhoeqpert(G[i, j].x, G[i, j].y, config)
            Q[i, j] = Data(dens, zero(eltype(dens)), zero(eltype(dens)))
        end
    end
end

function vortex2D!(Q::Matrix{Data}, G::Matrix{Grid}, config::SolverConfig)
    (; eps2) = config
    RealT = eltype(eps2)
    r1 = RealT(0.2)
    r2 = RealT(0.4)
    abar = RealT(0.1)
    a1 = abar / r1
    a2 = -abar * r2 / (r1 - r2)
    a3 = abar / (r1 - r2)

    for j = config.JBEG:config.JEND
        for i = config.IBEG:config.IEND

            r = sqrt((G[i, j].x - RealT(0.5))^2 + (G[i, j].y - RealT(0.5))^2)
            pphi = r^2
            if r <= r1
                u_radial = a1 * r
                integral = a1^2 * r^2 * 0.5f0
            elseif (r1 < r) && (r <= r2)
                u_radial = a2 + a3 * r
                integral =
                    a1^2 * r1^2 * 0.5f0 +
                    a2^2 * log(r / r1) +
                    a3^2 * (r^2 - r1^2) * 0.5f0 +
                    2 * a3 * a2 * (r - r1)
            else
                u_radial = RealT(0.0)
                integral =
                    a1^2 * r1^2 * 0.5f0 +
                    a2^2 * log(r2 / r1) +
                    a3^2 * (r2^2 - r1^2) * 0.5f0 +
                    2 * a3 * a2 * (r2 - r1)
            end

            den = RealT(1.0) + eps2 * RealT(0.5) * integral - RealT(0.5) * r^2
            vx = u_radial * (G[i, j].y - RealT(0.5)) / r
            vy = -u_radial * (G[i, j].x - RealT(0.5)) / r

            Q[i, j] = Data(den, den * vx, den * vy) # Conservative variables (density, momentum_x, momentum_y)
        end
    end
end

function boundary_calculation!(
    Q::Matrix{Data},
    G::Matrix{Grid},
    config::SolverConfig,
    BC::Int,
)
    (; boundary_reflective) = config
    if boundary_reflective
        for j = config.JBEG:config.JEND
            k = 1
            for i = config.NG:-1:1
                den = Q[i+k, j].DN
                vx = -Q[i+k, j].VX
                vy = Q[i+k, j].VY
                Q[i, j] = Data(den, vx, vy)
                k = k + 2
            end
        end
        for j = config.JBEG:config.JEND
            k = 1
            for i = (config.IEND+1):config.XNUM_TOT
                den = Q[i-k, j].DN
                vx = -Q[i-k, j].VX
                vy = Q[i-k, j].VY
                Q[i, j] = Data(den, vx, vy)
                k = k + 2
            end
        end

        for i = config.IBEG:config.IEND
            k = 1
            for j = config.NG:-1:1
                den = Q[i, j+k].DN
                vx = Q[i, j+k].VX
                vy = -Q[i, j+k].VY
                Q[i, j] = Data(den, vx, vy)
                k = k + 2
            end
        end
        for i = config.IBEG:config.IEND
            k = 1
            for j = (config.JEND+1):config.YNUM_TOT
                den = Q[i, j-k].DN
                vx = Q[i, j-k].VX
                vy = -Q[i, j-k].VY
                Q[i, j] = Data(den, vx, vy)
                k = k + 2
            end
        end
    elseif !boundary_reflective
        for j = config.JBEG:config.JEND
            for i = config.NG:-1:1
                den = Q[i+config.XNUM, j].DN
                vx = Q[i+config.XNUM, j].VX
                vy = Q[i+config.XNUM, j].VY
                Q[i, j] = Data(den, vx, vy)
            end
        end
        for j = config.JBEG:config.JEND
            for i = (config.IEND+1):config.XNUM_TOT
                den = Q[i-config.XNUM, j].DN
                vx = Q[i-config.XNUM, j].VX
                vy = Q[i-config.XNUM, j].VY
                Q[i, j] = Data(den, vx, vy)
            end
        end
        for i = config.IBEG:config.IEND
            for j = config.NG:-1:1
                den = Q[i, j+config.YNUM].DN
                vx = Q[i, j+config.YNUM].VX
                vy = Q[i, j+config.YNUM].VY
                Q[i, j] = Data(den, vx, vy)
            end
        end
        for i = config.IBEG:config.IEND
            for j = (config.JEND+1):config.YNUM_TOT
                den = Q[i, j-config.YNUM].DN
                vx = Q[i, j-config.YNUM].VX
                vy = Q[i, j-config.YNUM].VY
                Q[i, j] = Data(den, vx, vy)
            end
        end
    end
end

function fluxxt(Q::Data, config::SolverConfig)
    (; eps2, AP) = config
    P = pressure(Q, config) / eps2
    RealT = eltype(config.si)
    if AP
        den = RealT(0.0)
    else
        den = Q.VX
    end
    velx = Q.VX^2 / Q.DN + P
    vely = Q.VX * Q.VY / Q.DN
    return Data(den, velx, vely)
end

function fluxxpt(Q::Data, config::SolverConfig)
    P = pressure(Q) / eps2
    #return Data(0.0, P, 0.0)
    return Data(0.0, 0.0, 0.0)
end

function fluxyt(Q::Data, config::SolverConfig)
    (; eps2, AP) = config
    P = pressure(Q, config) / eps2
    RealT = eltype(config.si)
    if AP
        den = RealT(0.0)
    else
        den = Q.VY
    end
    velx = Q.VX * Q.VY / Q.DN
    vely = Q.VY^2 / Q.DN + P
    return Data(den, velx, vely)
end

function fluxypt(Q::Data, config::SolverConfig)
    (; eps2) = config
    RealT = eltype(eps2)
    P = pressure(Q, config) / eps2
    return Data(RealT(0.0), RealT(0.0), RealT(0.0))
end

function fluxx(Q::Data, config::SolverConfig)
    (; eps2) = config
    RealT = eltype(eps2)
    return Data(Q.VX, Q.DN / eps2, RealT(0.0))
end

function fluxy(Q::Data, config::SolverConfig)
    (; eps2) = config
    RealT = eltype(eps2)
    return Data(Q.VY, RealT(0), Q.DN / eps2)
end

function max_speedx(Q::Data, config)
    RealT = eltype(config.si)
    (; AP, GAMMA) = config
    if AP
        return RealT(2.0) * abs(Q.VX / Q.DN)
    else
        a = sqrt(GAMMA * Q.DN^(GAMAM1)) / eps
        return (max(Q.VX / Q.DN + a, Q.VX / Q.DN - a))
    end
end

function max_speedy(Q::Data, config)
    (; AP, GAMMA, GAMAM1, eps) = config
    RealT = eltype(config.si)
    if AP
        return RealT(2.0) * abs(Q.VY / Q.DN)
    else
        a = sqrt(GAMMA * Q.DN^(GAMAM1)) / eps
        return (max(Q.VY / Q.DN + a, Q.VY / Q.DN - a))
    end
end


function comp_phi_cc!(
    PHI::Matrix{RealT},
    G::Matrix{Grid},
    config::SolverConfig,
) where {RealT<:Real}
    (; phi) = config
    for j = config.JBEG:config.JEND
        for i = config.IBEG:config.IEND
            PHI[i, j] = phi(G[i, j].x, G[i, j].y)
        end
    end
    # boundary conditions   # same as density
    # left boundary
    for j = config.JBEG:config.JEND
        PHI[config.IBEG-1, j] = PHI[config.IBEG, j]
        PHI[config.IBEG-2, j] = PHI[config.IBEG+1, j]
    end

    # right boundary
    for j = config.JBEG:config.JEND
        PHI[config.IEND+1, j] = PHI[config.IEND, j]
        PHI[config.IEND+2, j] = PHI[config.IEND-1, j]
    end

    # bottom boundary
    for i = config.IBEG:config.IEND
        PHI[i, config.JBEG-1] = PHI[i, config.JBEG]
        PHI[i, config.JBEG-2] = PHI[i, config.JBEG+1]
    end

    # top boundary
    for i = config.IBEG:config.IEND
        PHI[i, config.JEND+1] = PHI[i, config.JEND]
        PHI[i, config.JEND+2] = PHI[i, config.JEND-1]
    end
end

function comp_h_cc!(
    H::Matrix{RealT},
    G::Matrix{Grid},
    config::SolverConfig,
) where {RealT<:Real}
    (; GMBGMM1, GMM1BGM, GMBGMM1, phi) = config
    for j = config.JBEG:config.JEND
        for i = config.IBEG:config.IEND
            H[i, j] = GMBGMM1 * log(GMM1BGM * (GMBGMM1 - phi(G[i, j].x, G[i, j].y)))
        end
    end
    # boundary conditions   # same as density
    # left boundary
    for j = config.JBEG:config.JEND
        H[config.IBEG-1, j] = H[config.IBEG, j]
        H[config.IBEG-2, j] = H[config.IBEG+1, j]
    end

    # right boundary
    for j = config.JBEG:config.JEND
        H[config.IEND+1, j] = H[config.IEND, j]
        H[config.IEND+2, j] = H[config.IEND-1, j]
    end

    # bottom boundary
    for i = config.IBEG:config.IEND
        H[i, config.JBEG-1] = H[i, config.JBEG]
        H[i, config.JBEG-2] = H[i, config.JBEG+1]
    end

    # top boundary
    for i = config.IBEG:config.IEND
        H[i, config.JEND+1] = H[i, config.JEND]
        H[i, config.JEND+2] = H[i, config.JEND-1]
    end
end



function source_calculation!(
    Q::Matrix{Data},
    G::Matrix{Grid},
    PHI::Matrix{RealT},
    H::Matrix{RealT},
    ESx::Matrix{RealT},
    ESy::Matrix{RealT},
    LSx::Matrix{RealT},
    LSy::Matrix{RealT},
    config::SolverConfig,
    WB::Int,
) where {RealT<:Real}

    fill!(ESx, RealT(0.0))
    fill!(ESy, RealT(0.0))
    fill!(LSx, RealT(0.0))
    fill!(LSy, RealT(0.0))
    (; HYDRORECON, eps2, AP, phi) = config
    for j = config.JBEG:config.JEND
        for i = config.IBEG:config.IEND

            if WB == 1
                if HYDRORECON
                    giph = 0.5 * (G[i+1, j].x + G[i, j].x)
                    gimh = 0.5 * (G[i, j].x + G[i-1, j].x)

                    gjph = 0.5 * (G[i, j+1].y + G[i, j].y)
                    gjmh = 0.5 * (G[i, j].y + G[i, j-1].y)

                    exphip1 = exp(H_eval(giph, G[i, j].y, config))
                    exphi = exp(H[i, j])
                    exphim1 = exp(H_eval(gimh, G[i, j].y, config))
                    exphjp1 = exp(H_eval(G[i, j].x, gjph, config))
                    exphjm1 = exp(H_eval(G[i, j].x, gjmh, config))

                    ESx[i, j] =
                        pressure(Q[i, j], config) * (exphip1 - exphim1) / (exphi * eps2)

                    ESy[i, j] =
                        pressure(Q[i, j], config) * (exphjp1 - exphjm1) / (exphi * eps2)
                else
                    if i == config.IBEG
                        ximh = RealT(0.5) * (G[i, j].x + G[i-1, j].x)
                        dxphip = phi(G[i+1, j].x, G[i+1, j].y) - phi(G[i, j].x, G[i, j].y)
                        dxphim = RealT(0.0)
                    elseif i == config.IEND
                        xiph = RealT(0.5) * (G[i, j].x + G[i+1, j].x)
                        dxphip = RealT(0.0)
                        dxphim = phi(G[i, j].x, G[i, j].y) - phi(G[i-1, j].x, G[i-1, j].y)
                    else
                        dxphip = phi(G[i+1, j].x, G[i+1, j].y) - phi(G[i, j].x, G[i, j].y)
                        dxphim = phi(G[i, j].x, G[i, j].y) - phi(G[i-1, j].x, G[i-1, j].y)
                    end

                    if j == config.JBEG
                        yimh = RealT(0.5) * (G[i, j].y + G[i, j-1].y)
                        dyphip = phi(G[i, j+1].x, G[i, j+1].y) - phi(G[i, j].x, G[i, j].y)
                        dyphim = RealT(0.0)
                    elseif j == config.JEND
                        yiph = RealT(0.5) * (G[i, j].y + G[i, j+1].y)
                        dyphip = RealT(0.0)
                        dyphim = phi(G[i, j].x, G[i, j].y) - phi(G[i, j-1].x, G[i, j-1].y)
                    else
                        dyphip = phi(G[i, j+1].x, G[i, j+1].y) - phi(G[i, j].x, G[i, j].y)
                        dyphim = phi(G[i, j].x, G[i, j].y) - phi(G[i, j-1].x, G[i, j-1].y)
                    end

                    rhobariph = (Q[i, j].DN + Q[i+1, j].DN) * 0.5f0
                    rhobarimh = (Q[i, j].DN + Q[i-1, j].DN) * 0.5f0

                    ESx[i, j] =
                        RealT(0.5) * (rhobariph * dxphip + rhobarimh * dxphim) / eps2

                    rhobarjph = (Q[i, j].DN + Q[i, j+1].DN) * 0.5f0
                    rhobarjmh = (Q[i, j].DN + Q[i, j-1].DN) * 0.5f0

                    ESy[i, j] =
                        RealT(0.5) * (rhobarjph * dyphip + rhobarjmh * dyphim) / eps2
                end

                if AP

                    # Linear source (for elliptic correction)
                    # For VX
                    if i == config.IBEG
                        rhoeqiph =
                            RealT(0.5) * (
                                rhoeq(G[i+1, j].x, G[i+1, j].y, config) +
                                rhoeq(G[i, j].x, G[i, j].y, config)
                            ) / eps2
                        rhoeqimh =
                            RealT(0.5) * (
                                rhoeq(G[i, j].x, G[i, j].y, config) +
                                rhoeq(G[i, j].x, G[i, j].y, config)
                            ) / eps2
                    elseif i == config.IEND
                        rhoeqiph =
                            RealT(0.5) * (
                                rhoeq(G[i, j].x, G[i, j].y, config) +
                                rhoeq(G[i, j].x, G[i, j].y, config)
                            ) / eps2
                        rhoeqimh =
                            RealT(0.5) * (
                                rhoeq(G[i, j].x, G[i, j].y, config) +
                                rhoeq(G[i-1, j].x, G[i-1, j].y, config)
                            ) / eps2
                    else
                        rhoeqiph =
                            RealT(0.5) * (
                                rhoeq(G[i+1, j].x, G[i+1, j].y, config) +
                                rhoeq(G[i, j].x, G[i, j].y, config)
                            ) / eps2
                        rhoeqimh =
                            RealT(0.5) * (
                                rhoeq(G[i, j].x, G[i, j].y, config) +
                                rhoeq(G[i-1, j].x, G[i-1, j].y, config)
                            ) / eps2
                    end
                    rhoBrhoeq = Q[i, j].DN / rhoeq(G[i, j].x, G[i, j].y, config)
                    LSx[i, j] = rhoBrhoeq * (rhoeqiph - rhoeqimh)


                    # For VY
                    if j == config.JBEG
                        rhoeqjph =
                            RealT(0.5) * (
                                rhoeq(G[i, j+1].x, G[i, j+1].y, config) +
                                rhoeq(G[i, j].x, G[i, j].y, config)
                            ) / eps2
                        rhoeqjmh =
                            RealT(0.5) * (
                                rhoeq(G[i, j].x, G[i, j].y, config) +
                                rhoeq(G[i, j].x, G[i, j].y, config)
                            ) / eps2
                    elseif j == config.JEND
                        rhoeqjph =
                            RealT(0.5) * (
                                rhoeq(G[i, j].x, G[i, j].y, config) +
                                rhoeq(G[i, j].x, G[i, j].y, config)
                            ) / eps2
                        rhoeqjmh =
                            RealT(0.5) * (
                                rhoeq(G[i, j].x, G[i, j].y, config) +
                                rhoeq(G[i, j-1].x, G[i, j-1].y, config)
                            ) / eps2
                    else
                        rhoeqjph =
                            RealT(0.5) * (
                                rhoeq(G[i, j+1].x, G[i, j+1].y, config) +
                                rhoeq(G[i, j].x, G[i, j].y, config)
                            ) / eps2
                        rhoeqjmh =
                            RealT(0.5) * (
                                rhoeq(G[i, j].x, G[i, j].y, config) +
                                rhoeq(G[i, j-1].x, G[i, j-1].y, config)
                            ) / eps2
                    end
                    LSy[i, j] = rhoBrhoeq * (rhoeqjph - rhoeqjmh)

                else
                    LSx[i, j] = 0.0
                    LSy[i, j] = 0.0
                end
            else    # Non-well balanced source
                if i == config.IBEG
                    ESx[i, j] =
                        -Q[i, j].DN * (
                            phi(G[i+1, j].x, G[i+1, j].y, config) -
                            phi(G[i, j].x, G[i, j].y, config)
                        ) / eps2
                elseif i == config.IEND
                    ESx[i, j] =
                        -Q[i, j].DN * (
                            phi(G[i, j].x, G[i, j].y, config) -
                            phi(G[i-1, j].x, G[i-1, j].y, config)
                        ) / eps2
                else
                    ESx[i, j] =
                        -0.5 *
                        Q[i, j].DN *
                        (
                            phi(G[i+1, j].x, G[i+1, j].y, config) -
                            phi(G[i-1, j].x, G[i-1, j].y, config)
                        ) / eps2
                end

                if j == config.JBEG
                    ESy[i, j] =
                        -Q[i, j].DN * (
                            phi(G[i, j+1].x, G[i, j+1].y, config) -
                            phi(G[i, j].x, G[i, j].y, config)
                        ) / eps2
                elseif j == config.JEND
                    ESy[i, j] =
                        -Q[i, j].DN * (
                            phi(G[i, j].x, G[i, j].y, config) -
                            phi(G[i, j-1].x, G[i, j-1].y, config)
                        ) / eps2
                else
                    ESy[i, j] =
                        -0.5 *
                        Q[i, j].DN *
                        (
                            phi(G[i, j+1].x, G[i, j+1].y, config) -
                            phi(G[i, j-1].x, G[i, j-1].y, config)
                        ) / eps2
                end
            end

        end
    end
end

function linear_recovery!(
    Q::Matrix{Data},
    Qx::Matrix{Data},
    Qy::Matrix{Data},
    config::SolverConfig,
)
    RealT = eltype(config.si)

    # normal recovery, just central differences
    for j = 2:config.YNUM_TOT-1
        for i = 2:config.XNUM_TOT-1
            Qx[i, j] = Data(
                RealT(0.5) * (Q[i+1, j].DN - Q[i-1, j].DN) / config.dx,
                RealT(0.5) * (Q[i+1, j].VX - Q[i-1, j].VX) / config.dx,
                RealT(0.5) * (Q[i+1, j].VY - Q[i-1, j].VY) / config.dx,
            )

            Qy[i, j] = Data(
                RealT(0.5) * (Q[i, j+1].DN - Q[i, j-1].DN) / config.dy,
                RealT(0.5) * (Q[i, j+1].VX - Q[i, j-1].VX) / config.dy,
                RealT(0.5) * (Q[i, j+1].VY - Q[i, j-1].VY) / config.dy,
            )
        end
    end
end

function flux_calculation!(
    Q::Matrix{Data},
    flxt::Matrix{Data},
    flyt::Matrix{Data},
    flx::Matrix{Data},
    fly::Matrix{Data},
    H::Matrix{RealT},
    G::Matrix{Grid},
    config::SolverConfig,
    dt::RealT,
    WB::Int,
) where {RealT<:Real}
    (; GAMMA, AP) = config
    # Initialize fluxes
    fill!(flxt, Data(RealT))
    fill!(flyt, Data(RealT))
    fill!(flx, Data(RealT))
    fill!(fly, Data(RealT))

    Qhydro = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)
    Qx = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)
    Qy = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)

    fill!(Qhydro, Data(RealT))
    fill!(Qx, Data(RealT))
    fill!(Qy, Data(RealT))

    if WB == 1
        # Hydrostatic substitution (interior)
        for j = config.JBEG:config.JEND
            for i = config.IBEG:config.IEND
                Qhydro[i, j] = Data(
                    p_hydro(Q[i, j].DN, G[i, j].x, G[i, j].y, config),
                    Q[i, j].VX,
                    Q[i, j].VY,
                )
            end
        end
        # Ghost cells: X boundaries (reflect VX, preserve VY)
        for j = config.JBEG:config.JEND
            for offset = 1:2
                il = config.IBEG - offset
                ir = config.IEND + offset
                Qhydro[il, j] =
                    Data(Q[il, j].DN^GAMMA * exp(-H[il, j]), Q[il, j].VX, Q[il, j].VY) #left
                Qhydro[ir, j] =
                    Data(Q[ir, j].DN^GAMMA * exp(-H[ir, j]), Q[ir, j].VX, Q[ir, j].VY) #right
            end
        end
        # Ghost cells: Y boundaries (reflect VY, preserve VX)
        for i = config.IBEG:config.IEND
            for offset = 1:2
                jd = config.JBEG - offset
                ju = config.JEND + offset
                Qhydro[i, jd] =
                    Data(Q[i, jd].DN^GAMMA * exp(-H[i, jd]), Q[i, jd].VX, Q[i, jd].VY) #bottom
                Qhydro[i, ju] =
                    Data(Q[i, ju].DN^GAMMA * exp(-H[i, ju]), Q[i, ju].VX, Q[i, ju].VY) #top
            end
        end
    else
        Qhydro .= Q
    end

    # reconstruction slopes
    linear_recovery!(Qhydro, Qx, Qy, config)

    for j = config.JBEG-1:config.JEND
        for i = config.IBEG-1:config.IEND
            # Reconstruction
            Qphmhydro = Data(
                Qhydro[i, j].DN + RealT(0.5) * Qx[i, j].DN * config.dx,
                Qhydro[i, j].VX + RealT(0.5) * Qx[i, j].VX * config.dx,
                Qhydro[i, j].VY + RealT(0.5) * Qx[i, j].VY * config.dx,
            )

            Qphphydro = Data(
                Qhydro[i+1, j].DN - RealT(0.5) * Qx[i+1, j].DN * config.dx,
                Qhydro[i+1, j].VX - RealT(0.5) * Qx[i+1, j].VX * config.dx,
                Qhydro[i+1, j].VY - RealT(0.5) * Qx[i+1, j].VY * config.dx,
            )

            if WB == 1
                # Back to primtive variables
                giph = RealT(0.5) * (G[i, j].x + G[i+1, j].x)

                # x-direction flux
                Qxphm = Data(
                    p_hydro_inv(Qphmhydro.DN, giph, G[i, j].y, config),
                    Qphmhydro.VX,
                    Qphmhydro.VY,
                )
                Qxphp = Data(
                    p_hydro_inv(Qphphydro.DN, giph, G[i, j].y, config),
                    Qphphydro.VX,
                    Qphphydro.VY,
                )
            else
                Qxphm = Data(Qphmhydro.DN, Qphmhydro.VX, Qphmhydro.VY)
                Qxphp = Data(Qphphydro.DN, Qphphydro.VX, Qphphydro.VY)
            end


            # Wave speeds
            a_lftx = max_speedx(Qxphm, config)
            a_rgtx = max_speedx(Qxphp, config)
            ax = max(a_lftx, a_rgtx)
            #ax = RealT(0.0)   # no diffusion 

            # Compute Explicit fluxes
            fluxxt_lft = fluxxt(Qxphm, config)
            fluxxt_rgt = fluxxt(Qxphp, config)

            if AP
                fluxxt_face = Data(
                    RealT(0.0),
                    RealT(0.5) *
                    (fluxxt_rgt.VX + fluxxt_lft.VX - ax * (Qxphp.VX - Qxphm.VX)),
                    RealT(0.5) *
                    (fluxxt_rgt.VY + fluxxt_lft.VY - ax * (Qxphp.VY - Qxphm.VY)),
                )
            else
                fluxxt_face = Data(
                    RealT(0.5) *
                    (fluxxt_rgt.DN + fluxxt_lft.DN - ax * (Qxphp.DN - Qxphm.DN)),
                    RealT(0.5) *
                    (fluxxt_rgt.VX + fluxxt_lft.VX - ax * (Qxphp.VX - Qxphm.VX)),
                    RealT(0.5) *
                    (fluxxt_rgt.VY + fluxxt_lft.VY - ax * (Qxphp.VY - Qxphm.VY)),
                )
            end

            # Update explicit fluxes at cell centers
            flxt[i, j] = Data(
                flxt[i, j].DN + fluxxt_face.DN,
                flxt[i, j].VX + fluxxt_face.VX,
                flxt[i, j].VY + fluxxt_face.VY,
            )

            # Compute Implicit Flux
            if AP
                fluxx_lft = fluxx(Q[i, j], config)
                fluxx_rgt = fluxx(Q[i+1, j], config)

                fluxx_face = Data(
                    RealT(0.5) * (fluxx_rgt.DN + fluxx_lft.DN),
                    RealT(0.5) * (fluxx_rgt.VX + fluxx_lft.VX),
                    RealT(0.5) * (fluxx_rgt.VY + fluxx_lft.VY),
                )

                # Update fluxes at cell centers
                flx[i, j] = Data(
                    flx[i, j].DN + fluxx_face.DN,
                    flx[i, j].VX + fluxx_face.VX,
                    flx[i, j].VY + fluxx_face.VY,
                )
            else
                fluxx_face = Data(0.0, 0.0, 0.0)
                flx[i, j] = Data(0.0, 0.0, 0.0)
            end

            if i < config.IEND
                flxt[i+1, j] = Data(
                    flxt[i+1, j].DN - fluxxt_face.DN,
                    flxt[i+1, j].VX - fluxxt_face.VX,
                    flxt[i+1, j].VY - fluxxt_face.VY,
                )

                flx[i+1, j] = Data(
                    flx[i+1, j].DN - fluxx_face.DN,
                    flx[i+1, j].VX - fluxx_face.VX,
                    flx[i+1, j].VY - fluxx_face.VY,
                )
            end
            # Reconstruction
            Qphmhydro = Data(
                Qhydro[i, j].DN + RealT(0.5) * Qy[i, j].DN * config.dy,
                Qhydro[i, j].VX + RealT(0.5) * Qy[i, j].VX * config.dy,
                Qhydro[i, j].VY + RealT(0.5) * Qy[i, j].VY * config.dy,
            )

            Qphphydro = Data(
                Qhydro[i, j+1].DN - RealT(0.5) * Qy[i, j+1].DN * config.dy,
                Qhydro[i, j+1].VX - RealT(0.5) * Qy[i, j+1].VX * config.dy,
                Qhydro[i, j+1].VY - RealT(0.5) * Qy[i, j+1].VY * config.dy,
            )

            if WB == 1
                # Back to primtive variables
                gjph = RealT(0.5) * (G[i, j].y + G[i, j+1].y)

                # y-direction flux
                Qyphm = Data(
                    p_hydro_inv(Qphmhydro.DN, G[i, j].x, gjph, config),
                    Qphmhydro.VX,
                    Qphmhydro.VY,
                )
                Qyphp = Data(
                    p_hydro_inv(Qphphydro.DN, G[i, j].x, gjph, config),
                    Qphphydro.VX,
                    Qphphydro.VY,
                )
            else
                Qyphm = Data(Qphmhydro.DN, Qphmhydro.VX, Qphmhydro.VY)
                Qyphp = Data(Qphphydro.DN, Qphphydro.VX, Qphphydro.VY)
            end

            # Compute Explicit fluxes
            fluxyt_tp = fluxyt(Qyphm, config)
            fluxyt_bt = fluxyt(Qyphp, config)

            # Wave speeds
            a_bty = max_speedy(Qyphm, config)
            a_tpy = max_speedy(Qyphp, config)

            ay = max(a_bty, a_tpy)
            #ay = RealT(0.0)   # no diffusion 

            if AP
                fluxyt_face = Data(
                    RealT(0.0),
                    RealT(0.5) * (fluxyt_tp.VX + fluxyt_bt.VX - ay * (Qyphp.VX - Qyphm.VX)),
                    RealT(0.5) * (fluxyt_tp.VY + fluxyt_bt.VY - ay * (Qyphp.VY - Qyphm.VY)),
                )

            else
                fluxyt_face = Data(
                    RealT(0.5) * (fluxyt_tp.DN + fluxyt_bt.DN - ay * (Qyphp.DN - Qyphm.DN)),
                    RealT(0.5) * (fluxyt_tp.VX + fluxyt_bt.VX - ay * (Qyphp.VX - Qyphm.VX)),
                    RealT(0.5) * (fluxyt_tp.VY + fluxyt_bt.VY - ay * (Qyphp.VY - Qyphm.VY)),
                )
            end

            # Update explicit fluxes at cell centers
            flyt[i, j] = Data(
                flyt[i, j].DN + fluxyt_face.DN,
                flyt[i, j].VX + fluxyt_face.VX,
                flyt[i, j].VY + fluxyt_face.VY,
            )

            # Compute Implicit Flux
            if AP
                fluxy_bt = fluxy(Q[i, j], config)
                fluxy_tp = fluxy(Q[i, j+1], config)

                fluxy_face = Data(
                    RealT(0.5) * (fluxy_tp.DN + fluxy_bt.DN),
                    RealT(0.5) * (fluxy_tp.VX + fluxy_bt.VX),
                    RealT(0.5) * (fluxy_tp.VY + fluxy_bt.VY),
                )

                # Update fluxes at cell centers
                fly[i, j] = Data(
                    fly[i, j].DN + fluxy_face.DN,
                    fly[i, j].VX + fluxy_face.VX,
                    fly[i, j].VY + fluxy_face.VY,
                )
            else
                fluxy_face = Data(0.0, 0.0, 0.0)
                fly[i, j] = Data(0.0, 0.0, 0.0)
            end

            if j < config.JEND
                flyt[i, j+1] = Data(
                    flyt[i, j+1].DN - fluxyt_face.DN,
                    flyt[i, j+1].VX - fluxyt_face.VX,
                    flyt[i, j+1].VY - fluxyt_face.VY,
                )
                fly[i, j+1] = Data(
                    fly[i, j+1].DN - fluxy_face.DN,
                    fly[i, j+1].VX - fluxy_face.VX,
                    fly[i, j+1].VY - fluxy_face.VY,
                )
            end
        end
    end
end

function rk_update!(
    Q0::Matrix{Data},
    Qh::Matrix{Data},
    flxt::Matrix{Data},
    flyt::Matrix{Data},
    flx::Matrix{Data},
    fly::Matrix{Data},
    ESx::Matrix{RealT},
    ESy::Matrix{RealT},
    LSx::Matrix{RealT},
    LSy::Matrix{RealT},
    config::SolverConfig,
    dt::RealT,
    c1::RealT,
    c2::RealT,
    G::Matrix{Grid},
) where {RealT<:Real}
    (; HYDRORECON, eps2) = config
    DELTAT1 = dt * c1
    DELTAT2 = dt * c2
    if HYDRORECON
        coeffE = 1.0
    else
        coeffE = -1.0
    end

    for j = config.JBEG:config.JEND
        for i = config.IBEG:config.IEND

            den =
                Q0[i, j].DN -
                DELTAT1 * (flxt[i, j].DN / config.dx + flyt[i, j].DN / config.dy) -
                DELTAT2 * (flx[i, j].DN / config.dx + fly[i, j].DN / config.dy)
            velx =
                Q0[i, j].VX -
                DELTAT1 * (
                    (flxt[i, j].VX - coeffE * ESx[i, j]) / config.dx +
                    flyt[i, j].VX / config.dy
                ) +
                DELTAT2 *
                ((flx[i, j].VX - LSx[i, j]) / config.dx + fly[i, j].VX / config.dy)

            vely =
                Q0[i, j].VY -
                DELTAT1 * (
                    flxt[i, j].VY / config.dx +
                    (flyt[i, j].VY - coeffE * ESy[i, j]) / config.dy
                ) +
                DELTAT2 *
                (flx[i, j].VY / config.dx + (fly[i, j].VY - LSy[i, j]) / config.dy)
            Qh[i, j] = Data(den, velx, vely)
        end
    end
end

function elliptic_solver!(
    Qh::Matrix{Data},
    Q0::Matrix{Data},
    G::Matrix{Grid},
    config::SolverConfig,
    DELTAT::RealT,
    an::RealT,
) where {RealT<:Real}
    (; eps2) = config
    if an == RealT(0.0)
        return
    end

    NX = config.XNUM
    NY = config.YNUM
    NUM = NX * NY

    dx = config.dx
    dy = config.dy

    cx = DELTAT^2 * an^2 / (dx^2 * eps2)
    cy = DELTAT^2 * an^2 / (dy^2 * eps2)

    rhs = Vector{RealT}(undef, NUM)
    for jl = 1:NY
        j = jl + config.NG
        for il = 1:NX
            i = il + config.NG
            k = (jl - 1) * NX + il
            rhs[k] = Qh[i, j].DN
        end
    end

    rows = Int[]
    cols = Int[]
    vals = RealT[]

    gidx(il, jl) = (jl - 1) * NX + il

    rho_eq_loc(il, jl) =
        rhoeq(G[il+config.NG, jl+config.NG].x, G[il+config.NG, jl+config.NG].y, config)

    for jl = 1:NY
        for il = 1:NX
            k = gidx(il, jl)

            if il < NX
                re_c = rho_eq_loc(il, jl)
                re_ip = rho_eq_loc(il + 1, jl)
                avg = RealT(0.5) * (re_c + re_ip)
                coeff_xp = cx * avg / re_c
                coeff_xp_nb = cx * avg / re_c

                jump = re_ip - re_c
                favg = RealT(0.5) * (re_ip + re_c)
                Ax = RealT(0.5) * jump / favg

                push!(rows, k)
                push!(cols, k)
                push!(vals, cx * (RealT(1.0) + Ax))
                push!(rows, k)
                push!(cols, gidx(il + 1, jl))
                push!(vals, cx * (RealT(-1.0) + Ax))
            end

            if il > 1
                re_c = rho_eq_loc(il, jl)
                re_im = rho_eq_loc(il - 1, jl)
                jump = re_c - re_im
                favg = RealT(0.5) * (re_c + re_im)
                Bx = RealT(0.5) * jump / favg

                push!(rows, k)
                push!(cols, k)
                push!(vals, cx * (RealT(1.0) - Bx))
                push!(rows, k)
                push!(cols, gidx(il - 1, jl))
                push!(vals, cx * (RealT(-1.0) - Bx))
            end

            if jl < NY
                re_c = rho_eq_loc(il, jl)
                re_jp = rho_eq_loc(il, jl + 1)
                jump = re_jp - re_c
                favg = RealT(0.5) * (re_jp + re_c)
                Ay = RealT(0.5) * jump / favg

                push!(rows, k)
                push!(cols, k)
                push!(vals, cy * (RealT(1.0) + Ay))
                push!(rows, k)
                push!(cols, gidx(il, jl + 1))
                push!(vals, cy * (RealT(-1.0) + Ay))
            end

            if jl > 1
                re_c = rho_eq_loc(il, jl)
                re_jm = rho_eq_loc(il, jl - 1)
                jump = re_c - re_jm
                favg = RealT(0.5) * (re_c + re_jm)
                By = RealT(0.5) * jump / favg

                push!(rows, k)
                push!(cols, k)
                push!(vals, cy * (RealT(1.0) - By))
                push!(rows, k)
                push!(cols, gidx(il, jl - 1))
                push!(vals, cy * (RealT(-1.0) - By))
            end

            push!(rows, k)
            push!(cols, k)
            push!(vals, RealT(1.0))
        end
    end

    M = sparse(rows, cols, vals, NUM, NUM)

    workspace = Problem(NUM, NUM, NUM, RealT(0))
    insparse!(workspace, M)
    infullrhs!(workspace, rhs)
    elliptic = SparseSolver(workspace)
    Sparspak.SparseSolver.solve!(elliptic)

    for jl = 1:NY
        j = jl + config.NG
        for il = 1:NX
            i = il + config.NG
            k = (jl - 1) * NX + il
            Qh[i, j] = Data(workspace.x[k], Qh[i, j].VX, Qh[i, j].VY)
            #Qh[i, j] = Data(Q0[i, j].DN, Qh[i, j].VX, Qh[i, j].VY)
        end
    end
end

# ============================================================================
# Velocity Update
# ============================================================================
function velocity_update!(
    Q::Matrix{Data},
    G::Matrix{Grid},
    LSx::Matrix{RealT},
    LSy::Matrix{RealT},
    config::SolverConfig,
    dt::RealT,
    an::RealT,
) where {RealT<:Real}
    cx = dt * an / config.dx
    cy = dt * an / config.dy
    (; eps2) = config
    for j = config.JBEG:config.JEND
        for i = config.IBEG:config.IEND
            rhoiph = RealT(0.5) * (Q[i+1, j].DN + Q[i, j].DN)
            rhoimh = RealT(0.5) * (Q[i, j].DN + Q[i-1, j].DN)

            rhojph = RealT(0.5) * (Q[i, j+1].DN + Q[i, j].DN)
            rhojmh = RealT(0.5) * (Q[i, j].DN + Q[i, j-1].DN)

            diffx = RealT(((rhoiph - rhoimh) / eps2 - LSx[i, j]))
            diffy = RealT(((rhojph - rhojmh) / eps2 - LSy[i, j]))

            Q[i, j] = Data(Q[i, j].DN, Q[i, j].VX - cx * diffx, Q[i, j].VY - cy * diffy)
        end
    end
end

function setup_tableau(IntegratorType::Int, RealT)
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
        # gm = 2.0 # monotone
        gm = 0.7 #1.0 / 3.0
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
    Q::Matrix{Data},
    G::Matrix{Grid},
    config::SolverConfig,
    filename::String = "output.dat",
)
    open(filename, "w") do fptr
        for j = config.JBEG:config.JEND
            for i = config.IBEG:config.IEND
                @printf(
                    fptr,
                    "%.5e\t%.5e\t%.22e\t%.22e\t%.22e\n",
                    Float64(G[i, j].x),
                    Float64(G[i, j].y),
                    Float64(Q[i, j].DN),
                    Float64(Q[i, j].VX),
                    Float64(Q[i, j].VY)
                )
            end
        end
    end
end

function write_sdata(
    Q::Matrix{RealT},
    G::Matrix{Grid},
    config::SolverConfig,
    filename::String = "output.dat",
) where {RealT<:Real}
    open(filename, "w") do fptr
        for j = config.JBEG:config.JEND
            for i = config.IBEG:config.IEND
                @printf(
                    fptr,
                    "%.5e\t%.5e\t%.22e\n",
                    Float64(G[i, j].x),
                    Float64(G[i, j].y),
                    Float64(Q[i, j])
                )
            end
        end
    end
end

function write_data_1D(
    Q::Matrix{RealT},
    G::Matrix{Grid},
    config::SolverConfig,
    filename::String = "output.dat",
) where {RealT<:Real}
    open(filename, "w") do fptr
        i = Int(config.XNUM / 2)
        for j = config.JBEG:config.JEND
            @printf(fptr, "%.5e\t%.22e\n", Float64(G[i, j].y), Float64(Q[i, j]))
        end
    end
end

function well_balance_check(
    Q::Matrix{Data},
    G::Matrix{Grid},
    config::SolverConfig,
    filename::String = "wb_check.dat",
)
    open(filename, "w") do fptr
        for j = config.JBEG:config.JEND
            for i = config.IBEG:config.IEND
                ddiff = sqrt((Q[i, j].DN - rhoeq(G[i, j].x, G[i, j].y))^2)
                velsq = sqrt(Q[i, j].VX^2 + Q[i, j].VY^2)

                @printf(
                    fptr,
                    "%.5e\t%.5e\t%.22e\t%.22e\n",
                    Float64(G[i, j].x),
                    Float64(G[i, j].y),
                    Float64(ddiff),
                    Float64(velsq)
                )
            end
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
    HYDRORECON,
    boundary_reflective = true,
    CFL,
)
    # Configuration
    # Run Parameters setup 
    # Initial conditions - choose one
    # "PROBLEM" ?  1 = well-balance, 2 = perturbed, 3 = 2D Vortex

    # "IntegratorType" ?    1 : DIRK(1,1,1), 2 : DP1A1,      3 : DP2A1, 
    #                       4 : DP2A2 ,      5 : ARS(2,2,2), 6 : DP-ARS(1,2,1)

    #final time
    #t_final = RealT(4.0)

    YNUMB = XNUMB
    RealT = eltype(eps)
    # Well-balanced? 0: NO, 1: Yes

    # Configuration
    config = create_config(
        XNUMB,
        YNUMB,
        t_final,
        eps,
        GAMMA,
        si,
        phi,
        AP,
        HYDRORECON,
        boundary_reflective,
        CFL,
    )
    println("Configuration created")
    println("XNUM = ", config.XNUM, ", dx = ", config.dx)
    println("YNUM = ", config.YNUM, ", dy = ", config.dy)

    println("eps = ", eps, ", eps2 = ", config.eps2)

    # Initialize grid
    G = initialize_grid(config)
    println("Grid initialized")

    #Boundary condition type
    #BC_TYPE  ? 1 : periodic, 2 : REFLECTIVE-boundary
    BC_TYPE = 2

    count = 0
    time = RealT(0.0)   # current time
    DELTAT = config.dx * factor # Initial time step

    # Allocate arrays
    Q = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)
    Q0 = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)
    Qx = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)
    Qy = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)

    Qs1 = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)
    Qs2 = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)
    Qs3 = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)
    Qs4 = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)

    Qunpe = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)
    Qdiff = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)

    flxt = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)
    flyt = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)

    flx = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)
    fly = Matrix{Data}(undef, config.XNUM_TOT, config.YNUM_TOT)

    ESx = Matrix{RealT}(undef, config.XNUM_TOT, config.YNUM_TOT)
    ESy = Matrix{RealT}(undef, config.XNUM_TOT, config.YNUM_TOT)
    LSx = Matrix{RealT}(undef, config.XNUM_TOT, config.YNUM_TOT)
    LSy = Matrix{RealT}(undef, config.XNUM_TOT, config.YNUM_TOT)

    PHI = Matrix{RealT}(undef, config.XNUM_TOT, config.YNUM_TOT)
    H = Matrix{RealT}(undef, config.XNUM_TOT, config.YNUM_TOT)

    MACH = Matrix{RealT}(undef, config.XNUM_TOT, config.YNUM_TOT)
    VORT = Matrix{RealT}(undef, config.XNUM_TOT, config.YNUM_TOT)

    DC = Vector{RealT}(undef, config.XNUM_TOT)
    GX = Vector{RealT}(undef, config.XNUM_TOT)

    fill!(Q, Data(RealT))
    fill!(Q0, Data(RealT))
    fill!(Qs1, Data(RealT))
    fill!(Qs2, Data(RealT))
    fill!(Qs3, Data(RealT))
    fill!(Qs4, Data(RealT))

    fill!(Qunpe, Data(RealT))
    fill!(Qdiff, Data(RealT))

    fill!(ESx, RealT(0.0))
    fill!(ESy, RealT(0.0))
    fill!(LSx, RealT(0.0))
    fill!(LSy, RealT(0.0))

    fill!(PHI, RealT(0.0))
    fill!(H, RealT(0.0))

    println("Arrays allocated")

    if PROBLEM == 1
        twod_well_balance!(Q0, G, config)
        println("Well-balanced initial condition set")

        well_balance_check(Q0, G, config, "wb_check_0.dat")
        println("Well-balance check written to wb_check_0.dat")
    elseif PROBLEM == 2
        twod_well_balance!(Qunpe, G, config)
        twod_pert_hydro!(Q0, G, config)
        println("Perturbed hydrostatic initial condition set")

        # print the initial density perturbation
        l2_ddiff = RealT(0.0)
        twod_pert_hydro!(Q0, G, config)
        for j = config.JBEG:config.JEND
            for i = config.IBEG:config.IEND
                den_diff = abs(Qunpe[i, j].DN - Q0[i, j].DN)
                Qdiff[i, j] = Data(den_diff, zero(eltype(den_diff)), zero(eltype(den_diff)))
            end
        end
        #write_data(Qdiff, G, config, "dendiff_0.dat")
        #println("\n Initial perturbation written to dendiff_0.dat")
    elseif PROBLEM == 3
        vortex2D!(Q0, G, config)
        println("2D vortex initial condition set")
    end

    boundary_calculation!(Q0, G, config, BC_TYPE)
    # Write output
    #write_data(Q0, G, config, "output_0.dat")
    #println("\n Initial data written to output_0.dat")

    # Open file for writing time evolve of balance
    if PROBLEM == 1
        name = @sprintf("time_balance.dat")
        fptr = open(name, "w")

        # Initialize accumulators
        l2_dn = RealT(0.0)
        l2_v = RealT(0.0)

        for j = config.JBEG:config.JEND
            for i = config.IBEG:config.IEND
                l2_dn += (Q0[i, j].DN - rhoeq(G[i, j].x, G[i, j].y))^2
                l2_v += Q0[i, j].VX^2 + Q0[i, j].VY^2
            end
        end
        # Compute RMS values
        ncells = config.XNUM * config.YNUM

        l2_dn = sqrt(l2_dn / ncells)
        l2_v = sqrt(l2_v / ncells)

        @show l2_dn
        @show l2_v
        # Write to file
        @printf(fptr, "%.5e\t%.24e\t%.24e\n", time, l2_dn, l2_v)
    end

    KE0 = kinetic_energy(Q0, G, config)
    if PROBLEM == 3
        # Open file for writing time evolve of balance
        name = @sprintf("results/kinetic_energy_%s_%s.dat", eps, XNUMB)
        gptr = open(name, "w")
        energy = kinetic_energy(Q0, G, config) / KE0
        # Write to file
        @printf(gptr, "%.5e\t%.24e\n", time, energy)
    end


    # Setup IMEX-RK tableau

    tableau = setup_tableau(IntegratorType, RealT)

    # Time stepping
    # time = RealT(0.0)

    println("\nStarting time integration...")
    println("Final time T = ", config.T)

    # Apply boundary conditions
    boundary_calculation!(Q0, G, config, BC_TYPE)

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

        @printf(
            "\nStep %d: DELTAT = %.6e, time = %.6e\n",
            count,
            Float64(DELTAT),
            Float64(time)
        )

        # Zero out fluxess
        fill!(flxt, Data(RealT))
        fill!(flyt, Data(RealT))
        fill!(flx, Data(RealT))
        fill!(fly, Data(RealT))

        # No flux calculation needed for first stage
        # Update explicit part of stage-1
        rk_update!(
            Q,
            Qs1,
            flxt,
            flyt,
            flx,
            fly,
            ESx,
            ESy,
            LSx,
            LSy,
            config,
            DELTAT,
            RealT(0.0),
            RealT(0.0),
            G,
        )
        boundary_calculation!(Qs1, G, config, BC_TYPE)

        if AP
            # Solve elliptic problem
            elliptic_solver!(Qs1, Q, G, config, DELTAT, tableau.a[1, 1])
            boundary_calculation!(Qs1, G, config, BC_TYPE)
            # Compute source terms
        end
        source_calculation!(Qs1, G, PHI, H, ESx, ESy, LSx, LSy, config, Well_balanced)

        if AP
            # Update momentum
            velocity_update!(Qs1, G, LSx, LSy, config, DELTAT, tableau.a[1, 1])
            boundary_calculation!(Qs1, G, config, BC_TYPE)
        end
        println(" Stage 1 completed ")

        # Flux calculation with stage-1 data
        flux_calculation!(Qs1, flxt, flyt, flx, fly, H, G, config, DELTAT, Well_balanced)    # reconstruction to be added later

        # Update explicit part of stage-2
        rk_update!(
            Q,
            Qs2,
            flxt,
            flyt,
            flx,
            fly,
            ESx,
            ESy,
            LSx,
            LSy,
            config,
            DELTAT,
            tableau.at[2, 1],
            tableau.a[2, 1],
            G,
        )
        boundary_calculation!(Qs2, G, config, BC_TYPE)

        if tableau.stages >= 3
            rk_update!(
                Q,
                Qs3,
                flxt,
                flyt,
                flx,
                fly,
                ESx,
                ESy,
                LSx,
                LSy,
                config,
                DELTAT,
                tableau.at[3, 1],
                tableau.a[3, 1],
                G,
            )
            #boundary_calculation!(Qs3, G, config, BC_TYPE)
        end

        if tableau.stages == 4
            rk_update!(
                Q,
                Qs4,
                flxt,
                flyt,
                flx,
                fly,
                ESx,
                ESy,
                LSx,
                LSy,
                config,
                DELTAT,
                tableau.at[4, 1],
                tableau.a[4, 1],
                G,
            )
            #boundary_calculation!(Qs4, G, config, BC_TYPE)
        end

        if AP
            # Solve elliptic problem
            elliptic_solver!(Qs2, Q, G, config, DELTAT, tableau.a[2, 2])
            boundary_calculation!(Qs2, G, config, BC_TYPE)
        end

        #Compute source terms
        source_calculation!(Qs2, G, PHI, H, ESx, ESy, LSx, LSy, config, Well_balanced)

        if AP
            # Update momentum
            velocity_update!(Qs2, G, LSx, LSy, config, DELTAT, tableau.a[2, 2])
            boundary_calculation!(Qs2, G, config, BC_TYPE)
        end

        println(" Stage 2 completed ")

        if tableau.stages == 2
            # Update solution
            Q .= Qs2
        end

        if tableau.stages >= 3
            # Update solution

            # Flux calculation with stage-2 data
            flux_calculation!(
                Qs2,
                flxt,
                flyt,
                flx,
                fly,
                H,
                G,
                config,
                DELTAT,
                Well_balanced,
            )    # reconstruction to be added later

            # Update explicit part of stage-3
            rk_update!(
                Qs3,
                Qs3,
                flxt,
                flyt,
                flx,
                fly,
                ESx,
                ESy,
                LSx,
                LSy,
                config,
                DELTAT,
                tableau.at[3, 2],
                tableau.a[3, 2],
                G,
            )
            boundary_calculation!(Qs3, G, config, BC_TYPE)

            if tableau.stages == 4
                rk_update!(
                    Qs4,
                    Qs4,
                    flxt,
                    flyt,
                    flx,
                    fly,
                    ESx,
                    ESy,
                    LSx,
                    LSy,
                    config,
                    DELTAT,
                    tableau.at[4, 2],
                    tableau.a[4, 2],
                    G,
                )
                boundary_calculation!(Qs4, G, config, BC_TYPE)
            end

            if AP
                # Solve elliptic problem
                elliptic_solver!(Qs3, Q, G, config, DELTAT, tableau.a[3, 3])
                boundary_calculation!(Qs3, G, config, BC_TYPE)
            end

            #Compute source terms
            source_calculation!(Qs3, G, PHI, H, ESx, ESy, LSx, LSy, config, Well_balanced)

            if AP
                # Update momentum
                velocity_update!(Qs3, G, LSx, LSy, config, DELTAT, tableau.a[3, 3])
                boundary_calculation!(Qs3, G, config, BC_TYPE)
            end

            println(" Stage 3 completed ")
        end
        if tableau.stages == 3
            # Update solution
            Q .= Qs3
        end

        if tableau.stages == 4

            # Flux calculation with stage-3 data
            flux_calculation!(
                Qs3,
                flxt,
                flyt,
                flx,
                fly,
                H,
                G,
                config,
                DELTAT,
                Well_balanced,
            )    # reconstruction to be added later

            # Update explicit part of stage-3
            rk_update!(
                Qs4,
                Qs4,
                flxt,
                flyt,
                flx,
                fly,
                ESx,
                ESy,
                LSx,
                LSy,
                config,
                DELTAT,
                tableau.at[4, 3],
                tableau.a[4, 3],
                G,
            )
            boundary_calculation!(Qs4, G, config, BC_TYPE)

            if AP
                # Solve elliptic problem
                elliptic_solver!(Qs4, Q, G, config, DELTAT, tableau.a[4, 4])
                boundary_calculation!(Qs4, G, config, BC_TYPE)
            end

            #Compute source terms
            source_calculation!(Qs4, G, PHI, H, ESx, ESy, LSx, LSy, config, Well_balanced)

            if AP
                # Update momentum
                velocity_update!(Qs4, G, LSx, LSy, config, DELTAT, tableau.a[4, 4])
                boundary_calculation!(Qs4, G, config, BC_TYPE)
            end

            println(" Stage 4 completed ")
            # Update solution
            Q .= Qs4
        end

        time += DELTAT
        if cfl_condition
            DELTAT = new_delt(Q, G, config)
        end
        if PROBLEM == 1
            l2_dn = RealT(0.0)
            l2_v = RealT(0.0)
            for j = config.JBEG:config.JEND
                for i = config.IBEG:config.IEND
                    l2_dn += (Q[i, j].DN - rhoeq(G[i, j].x, G[i, j].y))^2
                    l2_v += Q[i, j].VX^2 + Q[i, j].VY^2
                end
            end
            l2_dn = sqrt(l2_dn / ncells)
            l2_v = sqrt(l2_v / ncells)
            # Write to file
            @printf(fptr, "%.5e\t%.24e\t%.24e\n", time, l2_dn, l2_v)
        end

        if PROBLEM == 3
            # Open file for writing time evolve of balance
            energy = kinetic_energy(Q, G, config) / KE0
            # Write to file
            @printf(gptr, "%.5e\t%.24e\n", time, energy)
        end

    end
    #write_data(Q, G, config, "output_Q.dat")
    #println("\n Final solution written to output_Q.dat")

    if PROBLEM == 1
        @show l2_dn
        @show l2_v
    elseif PROBLEM == 2
        l2_ddiff = RealT(0.0)
        for j = config.JBEG:config.JEND
            for i = config.IBEG:config.IEND
                den_diff = abs(Qunpe[i, j].DN - Q[i, j].DN)
                Qdiff[i, j] = Data(den_diff, zero(eltype(den_diff)), zero(eltype(den_diff)))
            end
        end
        #write_data(Qdiff, G, config, "dendiff_Q.dat")
        #println("\n Final perturbation written to dendiff_Q.dat")
    elseif PROBLEM == 4
        # Initialize accumulators
        l2_dn = RealT(0.0)
        l2_vx = RealT(0.0)
        l2_vy = RealT(0.0)

        for j = config.JBEG:config.JEND
            for i = config.IBEG:config.IEND
                l2_dn += (Q0[i, j].DN - Q[i, j].DN)^2
                l2_vx += (Q0[i, j].VX - Q[i, j].VX)^2
                l2_vy += (Q0[i, j].VY - Q[i, j].VY)^2
            end
        end
        # Compute RMS values
        ncells = config.XNUM * config.YNUM

        l2_dn = sqrt(l2_dn / ncells)
        l2_vx = sqrt(l2_vx / ncells)
        l2_vy = sqrt(l2_vy / ncells)

        @show l2_dn
        @show l2_vx
        @show l2_vy
    elseif PROBLEM == 3 # VORTEX 2D
        for j = config.JBEG:config.JEND
            for i = config.IBEG:config.IEND
                soundspeed = GAMMA * pressure(Q[i, j], config) / Q[i, j].DN
                Mac = sqrt((Q[i, j].VX^2 + Q[i, j].VY^2) / soundspeed)
                MACH[i, j] = Mac
            end
        end
        #write_sdata(MACH, G, config, "MACH.dat")
        #println("\n Mach profile written to MACH.dat")

        # Compute vorticity 
        for j = config.JBEG:config.JEND
            for i = config.IBEG:config.IEND
                VORT[i, j] =
                    -0.125 * (
                        config.dx * (Q[i-1, j-1].VX + 2.0 * Q[i, j-1].VX + Q[i+1, j-1].VX) +
                        config.dy * (Q[i+1, j-1].VY + 2.0 * Q[i+1, j].VY + Q[i+1, j+1].VY) -
                        config.dx * (Q[i+1, j+1].VX + 2.0 * Q[i, j+1].VX + Q[i-1, j+1].VX) -
                        config.dy * (Q[i-1, j+1].VY + 2.0 * Q[i-1, j].VY + Q[i-1, j-1].VY)
                    ) / (config.dx * config.dy)
            end
        end
        write_data_1D(VORT, G, config, "results/vorticity_$(eps)_$(XNUMB).dat")
        println("\n Vorticity profile written to vorticity.dat")
    end
    return (; Q, G, Q0, Qunpe, Qdiff, MACH)
end


function phi_vortex(x::RealT, y::RealT) where {RealT<:Real}
    return RealT((x - RealT(0.5))^2 + (y - RealT(0.5))^2)
end

function phi_linear(x::RealT, y::RealT) where {RealT<:Real}
    return x + y
end

function plot_single(
    t_final,
    epsilon,
    si,
    HYDRORECON,
    h,
    index...;
    ticklabelsize = 13,
    levels = 13,
    factor = 0.1,
)
    RealT = Double64
    sol = main(
        PROBLEM = 2,
        IntegratorType = 4,
        t_final = t_final,
        XNUMB = 50,
        eps = epsilon,
        GAMMA = RealT(1.4),
        si = si,
        phi = phi_linear,
        AP = true,
        HYDRORECON = HYDRORECON,
        CFL = RealT(0.22),
        factor = RealT(factor),
        cfl_condition = false,
        boundary_reflective = true,
    )   # call main()
    XNUMB = 50
    NG = 2
    IBEG = 1 + NG
    IEND = XNUMB + NG
    JBEG = 1 + NG
    JEND = XNUMB + NG

    x_coords = [sol.G[i, j].x for i = IBEG:IEND, j = JBEG:JEND]
    y_coords = [sol.G[i, j].y for i = IBEG:IEND, j = JBEG:JEND]
    rho = [sol.Q[i, j].DN for i = IBEG:IEND, j = JBEG:JEND]
    rho0 = [sol.Qunpe[i, j].DN for i = IBEG:IEND, j = JBEG:JEND]
    value = abs.(rho .- rho0)
    c = contourf!(x_coords, y_coords, value, levels = levels, colormap = :cividis)
    Colorbar(h[index...], c, ticklabelsize = ticklabelsize)
end

function perturbed_plots()
    RealT = Double64
    t_final = RealT(0.05)
    epsilon = RealT(1)
    si = RealT(1e-1)

    h = Figure(size = (900, 700))
    labelsize = 22
    ticklabelsize = 13
    titlesize = 17

    kwargs = (
        xlabel = L"$x_1$",
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / 50 / 2, 1 - 1 / 50 / 2), (1 / 50 / 2, 1 - 1 / 50 / 2)),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        ylabel = L"$x_2$",
        titlesize = titlesize,
    )
    title = L"$T = 0.05$, WB-AP, $\varepsilon = 1$, $\zeta = 10^{-1}$"
    Axis(h[1, 1]; kwargs..., title = title)
    plot_single(t_final, epsilon, si, true, h, 1, 2)

    title = L"$T = 0.05$, NWB-AP, $\varepsilon = 1$, $\zeta = 10^{-1}$"
    Axis(h[1, 3]; kwargs..., title = title)
    plot_single(t_final, epsilon, si, false, h, 1, 4)

    si = RealT(1e-3)
    title = L"$T = 0.05$, WB-AP, $\varepsilon = 1$, $\zeta = 10^{-3}$"
    Axis(h[2, 1]; kwargs..., title = title)
    plot_single(t_final, epsilon, si, true, h, 2, 2)

    title = L"$T = 0.05$, NWB-AP, $\varepsilon = 1$, $\zeta = 10^{-3}$"
    Axis(h[2, 3]; kwargs..., title = title)
    plot_single(t_final, epsilon, si, false, h, 2, 4)

    save("results/plot_perturbed_1.png", h)

    h = Figure(size = (900, 400))
    title = L"$T = 0.005$, WB-AP, $\varepsilon = 10^{-1}$, $\zeta = 10^{-2}$"
    Axis(h[1, 1]; kwargs..., title = title)
    epsilon = RealT(1e-1)
    si = RealT(1e-2)
    t_final = RealT(0.005)
    plot_single(t_final, epsilon, si, true, h, 1, 2, factor = 0.001)

    title = L"$T = 0.001$, WB-AP, $\varepsilon = 10^{-2}$, $\zeta = 10^{-4}$"
    Axis(h[1, 3]; kwargs..., title = title)
    epsilon = RealT(1e-2)
    si = RealT(1e-4)
    t_final = RealT(0.001)
    plot_single(t_final, epsilon, si, true, h, 1, 4, factor = 0.001)

    save("results/plot_perturbed_2.png", h)
end

function plot_mach(
    t_final,
    epsilon,
    si,
    HYDRORECON,
    h,
    index...;
    ticklabelsize = 13,
    levels = 13,
    factor = 0.1,
    XNUMB = 30,
)
    RealT = Double64
    sol = main(
        PROBLEM = 3,
        IntegratorType = 4,
        t_final = t_final,
        XNUMB = XNUMB,
        eps = epsilon,
        GAMMA = RealT(2),
        si = si,
        phi = phi_vortex,
        AP = true,
        HYDRORECON = HYDRORECON,
        CFL = RealT(0.22),
        factor = RealT(factor),
        cfl_condition = true,
        boundary_reflective = true,
    )   # call main()

    NG = 2
    IBEG = 1 + NG
    IEND = XNUMB + NG
    JBEG = 1 + NG
    JEND = XNUMB + NG
    x_coords = [sol.G[i, j].x for i = IBEG:IEND, j = JBEG:JEND]
    y_coords = [sol.G[i, j].y for i = IBEG:IEND, j = JBEG:JEND]
    Mach = [sol.MACH[i, j] for i = IBEG:IEND, j = JBEG:JEND]
    c = contourf!(x_coords, y_coords, Mach, levels = levels, colormap = :cividis)
    Colorbar(h[index...], c, ticklabelsize = ticklabelsize)
end

function mach_plots()
    RealT = Double64
    epsilon = RealT(1)
    si = RealT(1e-1)

    h = Figure(size = (900, 700))
    labelsize = 22
    ticklabelsize = 24
    titlesize = 25
    XNUMB = 200
    kwargs = (
        xlabel = L"$x_1$",
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), (1 / XNUMB / 2, 1 - 1 / XNUMB / 2)),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        ylabel = L"$x_2$",
        titlesize = titlesize,
    )
    title = L"$T = 0.0$"
    t_final = RealT(0.0)
    Axis(h[1, 1]; kwargs..., title = title)
    plot_mach(
        t_final,
        epsilon,
        si,
        true,
        h,
        1,
        2,
        XNUMB = XNUMB,
        ticklabelsize = ticklabelsize,
    )

    save("results/mach_0.png", h)

    epsilon = RealT(1)

    h = Figure(size = (900, 700))
    labelsize = 22
    ticklabelsize = 20
    titlesize = 22
    t_final = RealT(1.0)
    epsilon = RealT(1e-1)
    kwargs = (
        xlabel = L"$x_1$",
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), (1 / XNUMB / 2, 1 - 1 / XNUMB / 2)),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        ylabel = L"$x_2$",
        titlesize = titlesize,
    )
    title = L"$T = 1.0$, $\varepsilon = 10^{-1}$"
    Axis(h[1, 1]; kwargs..., title = title)
    plot_mach(
        t_final,
        epsilon,
        si,
        true,
        h,
        1,
        2,
        XNUMB = XNUMB,
        ticklabelsize = ticklabelsize,
    )

    epsilon = RealT(1e-2)
    kwargs = (
        xlabel = L"$x_1$",
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), (1 / XNUMB / 2, 1 - 1 / XNUMB / 2)),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        ylabel = L"$x_2$",
        titlesize = titlesize,
    )
    title = L"$T = 1.0$, $\varepsilon = 10^{-2}$"
    Axis(h[1, 3]; kwargs..., title = title)
    plot_mach(
        t_final,
        epsilon,
        si,
        true,
        h,
        1,
        4,
        XNUMB = XNUMB,
        ticklabelsize = ticklabelsize,
    )

    epsilon = RealT(1e-3)
    kwargs = (
        xlabel = L"$x_1$",
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), (1 / XNUMB / 2, 1 - 1 / XNUMB / 2)),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        ylabel = L"$x_2$",
        titlesize = titlesize,
    )
    title = L"$T = 1.0$, $\varepsilon = 10^{-3}$"
    Axis(h[2, 1]; kwargs..., title = title)
    plot_mach(
        t_final,
        epsilon,
        si,
        true,
        h,
        2,
        2,
        XNUMB = XNUMB,
        ticklabelsize = ticklabelsize,
    )

    epsilon = RealT(1e-4)
    kwargs = (
        xlabel = L"$x_1$",
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), (1 / XNUMB / 2, 1 - 1 / XNUMB / 2)),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        ylabel = L"$x_2$",
        titlesize = titlesize,
    )
    title = L"$T = 1.0$, $\varepsilon = 10^{-4}$"
    Axis(h[2, 3]; kwargs..., title = title)
    plot_mach(
        t_final,
        epsilon,
        si,
        true,
        h,
        2,
        4,
        XNUMB = XNUMB,
        ticklabelsize = ticklabelsize,
    )
    save("results/mach_1.png", h)
end


function eps_to_str(eps)
    s = @sprintf("%.g", eps)
end

function load_dat(path)
    data = readdlm(path)
    return data[:, 1], data[:, 2]
end

function plot_vorticity_kinetic_energy()

    results_dir = "results"

    epsilons = [1e-1, 1e-2, 1e-3, 1e-4]

    fig = Figure(size = (1100, 450), fontsize = 13)

    labelsize = 22
    ticklabelsize = 24
    titlesize = 25
    kwargs = (
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        limits = ((1 / XNUMB / 2, 1 - 1 / XNUMB / 2), nothing),
        xticklabelsize = ticklabelsize,
        yticklabelsize = ticklabelsize,
        titlesize = titlesize,
    )
    ax_ke = Axis(fig[1, 1]; kwargs..., xlabel = L"t", title = L"Relative Kinetic Energy $$")

    ax_vor =
        Axis(fig[1, 2]; kwargs..., xlabel = L"x_1", title = L"Vorticity at $x_1 = 0.5$")

    colors = Makie.wong_colors()
    linestyles = [:solid, :dash, :dot, :dashdot]

    for (i, epsilon) in enumerate(epsilons)
        label = L"\varepsilon = 10^{%$(round(Int, log10(epsilon)))}"
        col = colors[i]
        estr = eps_to_str(epsilon)
        ls = linestyles[i]
        ke_file = joinpath(results_dir, "kinetic_energy_$(estr)_200.dat")
        t, ke = load_dat(ke_file)
        lines!(ax_ke, t, ke; color = col, linewidth = 2, label = label, linestyle = ls)

        vor_file = joinpath(results_dir, "vorticity_$(estr)_200.dat")
        x, vort = load_dat(vor_file)
        lines!(ax_vor, x, vort; color = col, linewidth = 2, label = label, linestyle = ls)
    end

    Legend(fig[1, 3], ax_ke, framevisible = true, labelsize = 25)

    save("results/plot_kinetic_vorticity.pdf", fig)
end
