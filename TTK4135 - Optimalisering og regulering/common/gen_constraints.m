function [vlb,vub] = gen_constraints(N,M,xl,xu,ul,uu)
% Function to generate constraints on states and inputs.
% N     - Time horizon for states
% M     - Time horizon for inputs
% xl    - Lower bound for states (column vector, mx*1)
% xu    - Upper bound for states (column vector, mx*1)
% ul    - Lower bound for inputs (column vector, mu*1)
% uu    - Upper bound for inputs (column vector, mu*1)
%
% Oppdatert 29/3-2001 av Geir Stian Landsverk
% Updated January 2018 by Andreas L. Flåten (translated to English)

vlb_x	= repmat(xl,N,1);
vub_x	= repmat(xu,N,1);

vlb_u	= repmat(ul,M,1);
vub_u	= repmat(uu,M,1);

vlb	    = [vlb_x; vlb_u];
vub	    = [vub_x; vub_u];


