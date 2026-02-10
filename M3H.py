import numpy as np
import copy

class Hypothesis:
    """
    A hypothesis for MMMH filter
    
    :var filter: endpoint filter of the hypothesis
    :vartype filter: KalmanFilter

    """
    def __init__(self, filter, modal_history: list, likelihood: float):
        self.filter  = filter 
        self.likelihood = likelihood
        self.modal_history = modal_history

    def __eq__(self, value: object) -> bool:
        pass

    def __repr__(self) -> str:
        return "Hypothesis object\n" +\
               f"Modal history: {self.modal_history}\n" +\
               f"Likelihood: {self.likelihood}"

class MMMH:
    """
    MMMH filter

    Args:
        filters: the filter bank
        Probas: markov chain matrix

        eps (float): pruning threshold
        L (int): merging depth
        l_max (int): maximum number of hypotheses
    
    Methods:

    """
    def __init__(self, filters, Probas, eps=0.01, L=3, l_max=3) -> None:
        if len(filters) < 2:
            raise ValueError("filters must contain at least two filters")
        
        self.filters = filters
        self.Probas = Probas
        self.eps = eps
        self.L = L
        self.l_max = l_max

        self.M = len(filters) # Число фильтров в банке

        init_likelihood = 1 / self.M 
        self.hypotheses = []

        ###
        for i in range(self.M):
            self.hypotheses.append( Hypothesis(filters[i], [i], init_likelihood) )
        ###


    def _expand(self) -> None:
        new_hypotheses = []

        # Expand l(k) hypos with l(k+1) child hypos
        # Take H and expand it.
        for H in self.hypotheses:
            i = H.modal_history[-1]
            for j, filt in enumerate(self.filters):

                # This assumes that all filter have the same dimensions and meanings for x and P
                new_filter = copy.deepcopy(filt)
                new_filter.x = H.filter.x
                new_filter.P = H.filter.P
                # BUT THIS DOES NOT ACCOUNT FOR THE MEASUREMENT THAT WAS USED LAST
                # THIS APPROACH MAKES USE OF ONLY THE FIRST MEASUREMENT
                new_filter.n = H.filter.n
                # there are also other values that may need copying
                new_filter.prior = H.filter.prior
                new_filter.P_prior = H.filter.P_prior
                # _count_parameters is out in the EKF for this one
                # TODO: make a better way initializing new endpoint filters for hypotheses

                H_new = Hypothesis(
                    filter=new_filter, # THIS WAS THE STARTING FILTER, IT DOESNT GET UPDATED!!!
                    modal_history=H.modal_history[:] + [j],
                    likelihood=self.Probas[i][j]*H.likelihood)
                # from i to j

                new_hypotheses.append(H_new)

        self.hypotheses = new_hypotheses

    def print_hypotheses_estimations(self) -> None:
        for H in self.hypotheses:
            print("----------")
            print(H)
            print(f"x = {H.filter.x}")

    def _merge(self) -> None:
        l = len(self.hypotheses)
        new_hypotheses = []
        merged_idx = []

        for k in range(l):
            if k not in merged_idx:
                suff1 = self.hypotheses[k].modal_history[-self.L:]

                new_hypotheses.append(Hypothesis(
                    filter=copy.deepcopy(self.hypotheses[k].filter),
                    modal_history=suff1,
                    #modal_history=self.hypotheses[k].modal_history[:],
                    likelihood=self.hypotheses[k].likelihood
                ))

                # to_merge_idx = []
                for m in range(k+1, l):
                    suff2 = self.hypotheses[m].modal_history[-self.L:]

                    if suff1 == suff2:
                        new_hypotheses[-1].likelihood += self.hypotheses[m].likelihood
                        merged_idx.append(m)
                        #print(f"Merged hypo {k} with {m}")
                        # remember to merge
                        #to_merge_idx.append(m)

                # print(f"Merging hypo {k} with {to_merge_idx}")
                # # merge hypotheses equal to kth
                # if len(to_merge_idx) > 0:
                #     new_likelihood = self.hypotheses[k].likelihood + \
                #     sum([self.hypotheses[idx].likelihood for idx in to_merge_idx])

                #     new_hypotheses.append(Hypothesis(
                #         filter=self.hypotheses[k].filter,
                #         modal_history=self.hypotheses[k].modal_history[-self.L:],
                #         likelihood=new_likelihood
                #     ))
                # else:
                #     # no matches for this hypo
                #     new_hypotheses.append(Hypothesis(
                #         filter=self.hypotheses[k].filter,
                #         modal_history=self.hypotheses[k].modal_history[-self.L:],
                #         likelihood=self.hypotheses[k].likelihood
                #     ))
                
        self.hypotheses = new_hypotheses

    def _prune(self) -> None:
        total_likelihood = sum([H.likelihood for H in self.hypotheses])

        new_hypotheses = []

        for H in self.hypotheses:
            if H.likelihood / total_likelihood >= self.eps:
                new_hypotheses.append(copy.deepcopy(H))
            else:
                #print(f"Removing hypothesis {H}")
                pass
            
        if len(new_hypotheses) > self.l_max:
            new_hypotheses = sorted(new_hypotheses, key=lambda H: H.likelihood, reverse=True)
            # just sort it then maybe

            new_hypotheses = new_hypotheses[:self.l_max]
            # note that this may discard hypotheses with equal likelihoods keeping only one of them

        self.hypotheses = new_hypotheses

    def predict(self) -> None:
        for H in self.hypotheses:
            H.filter.predict()
        

    def update(self) -> None:
        num_zeros = 0
        for H in self.hypotheses:
            # Update, step (7)
            H.filter.update()

            # Update likelihoods, step (6)
            #print(-H.filter.y.T @ np.linalg.inv(H.filter.S) @ H.filter.y)
            H.likelihood = np.exp(-0.5 * H.filter.y.T @ np.linalg.inv(H.filter.S) @ H.filter.y)
            #print('likelihood: ', H.likelihood)
            if H.likelihood <= 0:
                num_zeros += 1

            # If the exponent is practically zero, proceed with equal likelihoods
            if num_zeros == len(self.hypotheses):
                for H in self.hypotheses:
                    H.likelihood = 1


        # Normalize, step (8)
        total_likelihood = sum([H.likelihood for H in self.hypotheses])
        for H in self.hypotheses:
            H.likelihood = H.likelihood / total_likelihood
        
            
    def output(self):
        # This assumes that all filters have the same dimensions and meanings for x and P
        state = 0
        for H in self.hypotheses:
            state += H.likelihood * H.filter.x

        return state
        #return self.hypotheses[0].filter.x



    # def __repr__(self) -> str:
    #     return '\n'.join([
    #         'MMMH estimator object',
    #         'x', self.x,
    #         'P', self.P,
    #         'N', self.N,
    #         'mu', self.mu,
    #         'M', self.M,
    #         'cbar', self.cbar,
    #         'likelihood', self.likelihood,
    #         'omega', self.omega
    #         ])