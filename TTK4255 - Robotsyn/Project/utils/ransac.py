import numpy as np
import random as rand
        

class Ransac:
    def __init__(self, model, sample_size, threshold, min_model_votes, p=0.99, w=0.5):
        rand.seed()
        self.model = model
        self.data_size = model.n_data()
        self.sample_size = sample_size
        self.threshold = threshold
        self.min_model_votes = min_model_votes
        self.max_iterations = int(np.ceil(np.log(1-p) / np.log(1-w**sample_size))) # Could try estimating this while looping
    
    @property
    def __random_sample_idx(self):
        return np.array(rand.sample(range(self.data_size), k=self.sample_size))
        
    
    def run(self):
        """
        {{{
        Given:
            data - a set of observed data points
            model - a model that can be fitted to data points
            n - the minimum number of data values required to fit the model
            k - the maximum number of iterations allowed in the algorithm
            t - a threshold value for determining when a data point fits a model
            d - the number of close data values required to assert that a model fits well to data
        Return:
            bestfit - model parameters which best fit the data (or nil if no good model is found)
        iterations = 0
        bestfit = nil
        besterr = something really large
        while iterations < k {
            maybeinliers = n randomly selected values from data
            maybemodel = model parameters fitted to maybeinliers
            alsoinliers = empty set
            for every point in data not in maybeinliers {
                if point fits maybemodel with an error smaller than t
                    add point to alsoinliers
            }
            if the number of elements in alsoinliers is > d {
                % this implies that we may have found a good model
                % now test how good it is
                bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
                thiserr = a measure of how well model fits these points
                if thiserr < besterr {
                    bestfit = bettermodel
                    besterr = thiserr
                }
            }
            increment iterations
        }
        return bestfit
        }}}
        """
        i = 1
        best_fit = None
        best_err = np.Inf
        while i < self.max_iterations:
            sample_idx = self.__random_sample_idx
            voters_idx = np.delete(np.arange(self.data_size), sample_idx)
            model_proposal = self.model.estimate_model_params(sample_idx)
            pot_inliers_idx = voters_idx[self.model.data_fit_error(model_proposal, voters_idx) < self.threshold]
            num_pot_inliers = len(pot_inliers_idx)
            if num_pot_inliers >= self.min_model_votes:
                improved_sample_idx = np.block([sample_idx, pot_inliers_idx])
                improved_model_proposal = self.model.estimate_model_params(improved_sample_idx)
                # TODO: Could make this heuristic more fancy
                improved_err = np.mean(self.model.data_fit_error(improved_model_proposal, improved_sample_idx))
                if improved_err < best_err:
                    best_fit = improved_model_proposal
                    best_err = improved_err
                    best_data_idx = improved_sample_idx
            i += 1
        if best_fit is None:
            raise ValueError("did not meet fit acceptance criteria")
        return best_fit, best_err, improved_sample_idx