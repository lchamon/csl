# -*- coding: utf-8 -*-
"""Factory for resilient constrained learning solvers

"""


def Resilient(PrimalDualSolver):
    """Factory method for resilient primal-dual solvers.

    Takes a primal-dual solver and wraps a resilient step to update
    specifications according to a resilient cost. See `ResilientSolver`
    for additional informations.

    ..warning:: Use at your own risk

    Parameters
    ----------
    PrimalDualSolver : `type`
        A primal-dual solver class inheriting from `PrimalDualBase`.

    Returns
    -------
    ResilientPrimalDualSolver: `type`
        A class for the resilient version of ``PrimalDualSolver``.

    """
    class ResilientSolver(PrimalDualSolver):
        """Resilient solver.

        Run ``PrimalDualSolver`` then update specifications either implicitly using
        a counterfactual update or explicitly using a resilience cost.

        If ``counterfactual`` is defined, then it is used to directly update
        the constraint specifications. Equivalent to defining a ``resilience_cost``
        if ``counterfactual`` evaluates the inverse of the gradient of ``resilience_cost``.

        If ``counterfactual`` is undefined or `None`, then an explicit ``resilience_cost``
        must be defined together with a solver to indirectly compute the
        resilient specifications using a fixed-point dynamics.

        Attributes
        ----------
        counterfactual : `callable`
            Evaluate the counterfactual slacks. Takes arguments

            - ``lambdas``: Dual variables corresponding to the average constraints (`list` [`torch.tensor`, (1, )])
            - ``mus``: Dual variables corresponding to the pointwise constraints (`list` [`torch.tensor`, (N, )])

            and returns

            - ``rhs``: Updated specification corresponding to the average constraints (`list` [`torch.tensor`, (1, )])
            - ``pointwise_rhs``: Updated specification corresponding to the pointwise constraints (`list` [`torch.tensor`, (N, )])

        resilience_cost : `callable`
            Evaluate the resilience cost and gradient. Takes arguments

            - ``lambdas``: Dual variables corresponding to the average constraints (`list` [`torch.tensor`, (1, )])
            - ``mus``: Dual variables corresponding to the pointwise constraints (`list` [`torch.tensor`, (N, )])

            and returns

            - ``h``: Resilience cost (`torch.tensor`, (1, ) with ``requires_grad=True``)

        resilience_solver : `torch.optim`
            Resilience problem solver

        resilience_step_size : `torch.optim.lr_scheduler`
            Resilience problem step size scheduler

        See also
        --------
        `PrimalDualBase` for additional attributes.

        """
        __name__ = f'Resilient{PrimalDualSolver.__name__}'
        __qualname__ = f'Resilient{PrimalDualSolver.__qualname__}'

        def __init__(self, user_settings={}):
            """Resilient solver constructor

            Parameters
            ----------
            user_settings : `dict`, optional
                Dictionary containing solver settings. See original solver class
                for basic solver settings and defaults. Additional specific settings:

                - ``counterfactual``: A function that evaluates the counterfactual slacks.
                  Implements the inverse of the gradient of the resilience cost.
                  Uses ``resilience_cost`` if `None`.
                - ``resilience_cost``: A function that evaluates the resilience cost.
                - ``resilience_solver``: Constructor for the PyTorch solver used to
                  solve the resilience problem (takes parameter, returns torch.optim).
                  The default is ADAM.
                - ``lr_r0``: Initial resilient step size. The default is 0.1.
                - ``lr_r_scheduler``: Resilient step size scheduler. The default is `None` (no decay).

            """
            # Pop resilient solver settings from base solver settings
            self.counterfactual = user_settings.pop('counterfactual', None)
            self.resilience_cost = user_settings.pop('resilience_cost', None)
            resilience_solver = user_settings.pop('resilience_solver', None)
            lr_r0 = user_settings.pop('lr_r0', None)
            lr_r_scheduler = user_settings.pop('lr_r_scheduler', None)

            super().__init__(user_settings)

            # Include resilient solver settings in global solver settings:
            # 'resilience_solver', 'lr_r0', 'lr_r_scheduler'
            self.settings.global_settings['resilience_solver'] = resilience_solver
            self.settings.global_settings['lr_r0'] = lr_r0
            self.settings.global_settings['lr_r_scheduler'] = lr_r_scheduler


            # Add resilient solver and step-size scheduler
            self.resilience_solver = None
            self.resilience_step_size = None

        def primal_dual_update(self, problem):
            # Call base primal-dual update
            primal_value_est, primal_grad_norm_est, \
                constraint_slacks_est, pointwise_slacks_est, \
                    dual_grad_norm_est = super().primal_dual_update(problem)

            ### RESILIENT STEP ###
            if self.counterfactual is not None:
                rhs, pointwise_rhs = self.counterfactual(problem.lambdas, problem.mus)

                for ii, r in enumerate(rhs):
                    problem.rhs[ii] = r

                for ii, r in enumerate(pointwise_rhs):
                    problem.pointwise_rhs[ii] = r
            else:
                # Set gradients
                with self._grads(problem.rhs + problem.pointwise_rhs):
                    problem.rhs.zero_grad()
                    problem.pointwise_rhs.zero_grad()
                    h = self.resilience_cost(problem.rhs, problem.pointwise_rhs)
                    h.backward()

                for ii, lambda_value in enumerate(problem.lambdas):
                    problem.rhs[ii].grad -= lambda_value
                    primal_grad_norm_est += problem.rhs[ii].grad.norm().item()**2

                for ii, mu_value in enumerate(problem.mus):
                    problem.pointwise_rhs[ii].grad -= mu_value
                    primal_grad_norm_est += problem.pointwise_rhs[ii].grad.norm().item()**2

                # Gradient gradient step
                self.resilience_solver.step()
                self.state_dict['resilience_solver'] = self.primal_solver.state_dict()

                # Update dual step size
                if self.resilience_step_size is not None:
                    self.resilience_step_size.step()

            return primal_value_est, primal_grad_norm_est, constraint_slacks_est, pointwise_slacks_est, dual_grad_norm_est


        def reset(self):
            super().reset()
            self.resilience_solver = None
            self.resilience_step_size = None


        def _initialize(self, problem):
            # Call to base class initialization
            super()._initialize(problem)

            # Initialize resilience solver
            if self.settings['resilience_solver'] is not None:
                self.resilience_solver = self.settings['resilience_solver'](problem.rhs + problem.pointwise_rhs,
                                                                            lr=self.settings['lr_r0'])

            # Load previous settings
            if 'resilience_solver' in self.state_dict:
                self.resilience_solver.load_state_dict(self.state_dict['resilience_solver'])

            # Initialize resilience step size scheduler
            if self.settings['lr_r_scheduler'] is not None:
                self.resilience_step_size = self.settings['lr_r_scheduler'](self.resilience_solver)


        class _grads:
            def __init__(self, variables):
                self.variables = variables

            def __enter__(self):
                for ii, _ in enumerate(self.variables):
                    self.variables[ii].requires_grad = True

            def __exit__(self, *args):
                for ii, _ in enumerate(self.variables):
                    self.variables[ii].requires_grad = False


    return ResilientSolver
