#In this file, we give some definition of functions, which could be used to test our algorithm
using ForwardDiff
using LinearAlgebra

function possion_1d(real_solution::Function)
    #@parameters x
    #@variables u(..)
    u(x) = real_solution(x)
    u_grad(x) = ForwardDiff.derivative(x -> u(x),x)
    g_1(x) = -ForwardDiff.derivative(x -> u_grad(x),x)
    g_2(x) = real_solution(x)
    return u,g_1,g_2
end

function cos_v(ν)
    function cos_vx(x)
        return cos(ν*x)
    end
    return cos_vx
end

function constant(a)
    function a_func(x)
        return (-a/2)*x^2
    end
    return a_func
end



function possion_2d(real_solution::Function)
    #@parameters x,y
    #@variables u(..)
    u(x,y) = real_solution(x,y)
    u_x(x,y) = ForwardDiff.derivative(x -> u(x,y),x)
    u_xx(x,y) = ForwardDiff.derivative(x -> u_x(x,y),x)
    u_y(x,y) = ForwardDiff.derivative(y -> u(x,y),y)
    u_yy(x,y) = ForwardDiff.derivative(y-> u_y(x,y),y)
    g_1(x,y) = -(u_xx(x,y)+u_yy(x,y))
    g_2(x,y) = real_solution(x,y)
    return u,g_1,g_2
end

function cos_v_2d(ν)
    function cos_v_xy(x,y)
        return cos(ν*(x+y))
    end
    return cos_v_xy
end

function constant_2d(a)
    function a_func_2d(x,y)
        return (-a/4)*x^2+(-a/4)*y^2
    end
    return a_func_2d
end