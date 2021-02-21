import torch
import numpy as np

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        # assert len(result) == 3
        # assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            val_auc_argmax = result[:, 5].argmax().item()
            val_ap_argmax = result[:, 6].argmax().item()

            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
            print(f'   Final AUC Test: {result[val_auc_argmax, 3]:.2f}')
            print(f'   Final AP Test: {result[val_ap_argmax, 4]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()

                val_auc_argmax = r[:, 5].argmax()
                test_auc_argmx = r[:, 6].argmax()

                test1 = r[val_auc_argmax, 3].item()
                test2 = r[test_auc_argmx, 4].item()
                best_results.append((train1, valid, train2, test, test1, test2))

            best_result = torch.tensor(best_results)
            write_result = []
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            write_result.append(str(np.around(r.mean().item(),2))+'+'+str(np.around(r.std().item(),2)))
            r = best_result[:, 4]
            print(f'   Final AUC Test: {r.mean():.2f} ± {r.std():.2f}')
            write_result.append(str(np.around(r.mean().item(),2))+'+'+str(np.around(r.std().item(),2)))
            r = best_result[:, 5]
            print(f'   Final AP Test: {r.mean():.2f} ± {r.std():.2f}')
            write_result.append(str(np.around(r.mean().item(),2))+'+'+str(np.around(r.std().item(),2)))
            return write_result