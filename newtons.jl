# Maintainer:   Thomas Kellogg
# References:   https://docs.juliaplots.org/stable/
# License:      https://opensource.org/license/mit

using Pkg
Pkg.build("PyCall")

using MTH229
using Surrogates
using SurrogatesPolyChaos
using Plots
# GLobal Initializations
iteration = 1;
array_out = []
# Implement Rosenbrock function and its gradient/hessian
function f(x)
    f = (1-x[1])^2 + 100 * (x[2] - x[1]^2)^2
    return f
end

function g(x)
    g1 = -2(1-x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    g2 = 200 * (x[2] - x[1]^2)
    g = [g1,g2]
    return g
end

function h(x)
    h11 = 1200 * x[1]^2 - 400 * x[2] + 2
    h12 = -400 * x[1]
    h21 = h12
    h22 = 200
    h = [h11 h12; h21 h22]
    return h
end

function push_to_array(o)
    push!(array_out, o)
end

function call_array()
    return array_out
end
# implement Armijo's Line Search
function armijo_line_search(x, gk, p, cost_old, alpha)
    # init
    K = 1000
    k = 0
    descent = 0
    c = 10^(-4)
    rho = 0.5

    xtry = x + alpha * p
    # search for converging value of alpha
    while (k < K) & (descent == 0)

        xtry = x + alpha * p
        cost = f(xtry)
        # check if descent success
        if cost < cost_old + c * alpha * dot(gk, p)
            cost_old = cost
            descent = 1
        else    # otherwise continue
            k = k + 1
            alpha = rho * alpha
        end
        println("\tIter: ",k ," / ", K, "\t cost: ", cost, "\t alpha: ", alpha)
    end
    
    if descent == 0
        println("Linesearch failed; no descent after ", k, " steps") 
    end

    return alpha
end
# implement Newton's Method
function newtons_method(x, alpha, tol)
    # init
    K = 1000
    hk = h(x)
    gk = g(x)
    o = [x[1],x[2], alpha]
    push_to_array(o)
    # convergence search
    for item in 1:K
         # check if we are within norm tolerance    
        if norm(inv(hk) * gk) < tol
            println("Newton's Method succeeded in ", item, " steps.")
            global iteration = item
            return
        end
        
        println("Step: ", item, "\t| x: ", x, "\t| alpha: ", alpha)

        gk = g(x)
        hk = h(x)
        p = -1*(inv(hk) * gk)
        # Line Search
        alpha = armijo_line_search(x, gk, p, f(x), alpha)
        x = x + alpha * p

        o = [x[1],x[2], alpha]
        push_to_array(o)   
    end
    println("Newton's Method FAILURE in ", K, " steps.")
    return

end

# animation creation function
function createGif(alpha)
    # initialize plot
    plt = plot3d(
        1,
        xlim = (-30, 30),
        ylim = (-30, 30),
        zlim = (0, alpha + 1),
        title = "Convergence with Newton's Method",
        xlabel = "x",
        ylabel = "y",
        zlabel = "alpha",
        legend = false
    )
    array_x = call_array()
    # render gif
    @gif for i=1:iteration
        array_x = call_array()
        xval = array_x[i][1][1]
        yval = array_x[i][2][1]
        aval = array_x[i][3][1]
        push!(plt, xval, yval, aval)
        
    end
end

x = [5,1]
x = float.(x)
tol = 10^(-4)
maxiter = 5
alpha = 5

newtons_method(x, alpha, tol)

createGif(alpha)
