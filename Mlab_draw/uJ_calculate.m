function [uJ] = uJ_calculate(uN, PN, PJ)
    global c_f c_o;
    uJ = -sum(uN) - c_f * sum(PN) - c_o * sum(PJ);
end