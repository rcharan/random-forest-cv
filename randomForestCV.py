import sklearn.model_selection as cv
import sklearn.ensemble        as ensem

class RandomForestClassifierCV(ensem.RandomForestClassifier):
    '''
        This class implements cross validation *ONLY* for the 
        number of estimators. It does this using OOB scoring and
        warm starts. In theory this should be doable in sk-learn by
        default, but I am unable to make it work without this
        roll-you-own solution
        
        Note: always computes up to max_estimators. Then picks the
        smallest possible within the tolerance. There are presumably
        better ways to detect stabilization.
        
        For other hyperparemeters, use GridSearch, below
    '''
    
    def __init__(self, n_estimators_range = None, tolerance = 0.005, **kwargs):
        kwargs['warm_start']   = True
        kwargs['oob_score']    = True
        super().__init__(**kwargs)
        self.tolerance            = tolerance
        self.use_oob_score        = True
        self.init_kwargs          = kwargs
        
        if n_estimators_range is None:
          n_estimators_range = list(range(100, 1000 + 1, 100))
        self.n_estimators_range   = n_estimators_range
        
    def reset_and_clone(self):
        return RandomForestClassifierCV(n_estimators   = self.n_estimators,
                                        tolerance      = self.tolerance, 
                                       **self.init_kwargs)
        
    def score(self, *args, **kwargs):
        if self.use_oob_score:
            return self.best_score
        else:
            return super().score(*args, **kwargs)
    
    def fit(self, X, y):
        self.n_est_scores_ = {}
        for i in self.n_estimators_range:
            self.n_estimators = i
            super().fit(X, y)
            self.n_est_scores_[i] = self.oob_score_
            
        self.best_score = max(self.n_est_scores_.values())
        self.best_n     = next(n for n, v in self.n_est_scores_.items() if v > self.best_score - self.tolerance)
            
        return self
    
    def finalize(self, X, y):
        self.use_oob_score = False
        
        self.set_params(warm_start = False)
        
        self.n_estimators = self.best_n
        super().fit(X, y)
        
    def get_params(self, *args, **kwargs):
        out = ensem.RandomForestClassifier().get_params()
        out['n_estimators_range'] = self.n_estimators_range
        out['tolerance']      = self.tolerance
        return out


class GridSearch():
    '''
        Grid Search *without* cross validation.
        Doesn't implement full feature set of GridSearch
        
        Just fit, predict in a bare-bones fashion
        
        Designed for use with RandomForestCV above
    '''
    
    def __init__(self, estimator, param_grid, verbose = 0):
        self.param_grid       = cv.ParameterGrid(param_grid)
        self.parent_estimator = estimator
        self.verbose          = verbose
        
    def fit(self, X, y):
        self.best_params = None
        self.best_model  = None
        self.best_score  = 0
        
        progress_count   = 0
        if self.verbose:
            print(f'{len(self.param_grid)} Iterations to do')
        
        for param_dict in self.param_grid:
            model = self.parent_estimator.reset_and_clone()
            model.set_params(**param_dict)
            model.fit(X, y)
            
            if model.score() > self.best_score:
                self.best_params = param_dict
                self.best_model  = model
                self.best_score  = model.score()
                
            progress_count += 1
            if self.verbose and progress_count % 20 == 0:
                print(f'{progress_count} Done')
        
        # This fails if every model somehow had an accuracy of 0...
        assert bool(self.best_params)
        
        self.best_model.finalize(X, y)
        
        return self.best_model
        
    def predict(self, *args, **kwargs):
        return self.best_model.predict(*args, **kwargs)
    
    def score(self, *args, **kwargs):
        return self.best_model.score(*args, **kwargs)