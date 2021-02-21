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
            pass
            # result = 100 * torch.tensor(self.results[run])
            # argmax = result[:, 1].argmax().item()
            # val_auc_argmax = result[:, 5].argmax().item()
            # val_ap_argmax = result[:, 6].argmax().item()

            # print(f'Run {run + 1:02d}:')
            # print(f'Highest Train: {result[:, 0].max():.2f}')
            # print(f'Highest Valid: {result[:, 1].max():.2f}')
            # print(f'  Final Train: {result[argmax, 0]:.2f}')
            # print(f'   Final Test: {result[argmax, 2]:.2f}')
            # print(f'   Final AUC Test: {result[val_auc_argmax, 3]:.2f}')
            # print(f'   Final AP Test: {result[val_ap_argmax, 4]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)
            # result_name = ['acc', 'f1_mac', 'prec_mac', 'f1_mic', 'prec_mic', 'nmi', 'adjscore']

            best_results = []
            for r in result:

                acc = r[0,0].item()
                f1_mac = r[0,1].item()
                prec_mac = r[0,2].item()
                f1_mic = r[0,3].item()
                prec_mic = r[0,4].item()
                nmi = r[0,5].item()
                adjscore = r[0,6].item()

                best_results.append((acc, f1_mic, f1_mac, prec_mic, prec_mac, nmi, adjscore))

            best_result = torch.tensor(best_results)
            write_result = []
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Mean Acc: {r.mean():.2f} ± {r.std():.2f}')
            write_result.append(str(np.around(r.mean().item(),2))+'+'+str(np.around(r.std().item(),2)))

            r = best_result[:, 1]
            print(f'Mean f1_mic: {r.mean():.2f} ± {r.std():.2f}')
            write_result.append(str(np.around(r.mean().item(),2))+'+'+str(np.around(r.std().item(),2)))

            r = best_result[:, 2]
            print(f'Mean f1_mac: {r.mean():.2f} ± {r.std():.2f}')
            write_result.append(str(np.around(r.mean().item(),2))+'+'+str(np.around(r.std().item(),2)))

            r = best_result[:, 3]
            print(f'Mean prec_mic: {r.mean():.2f} ± {r.std():.2f}')
            write_result.append(str(np.around(r.mean().item(),2))+'+'+str(np.around(r.std().item(),2)))
            r = best_result[:, 4]
            print(f'Mean prec_mac: {r.mean():.2f} ± {r.std():.2f}')
            write_result.append(str(np.around(r.mean().item(),2))+'+'+str(np.around(r.std().item(),2)))
            r = best_result[:, 5]
            print(f'Mean nmi: {r.mean():.2f} ± {r.std():.2f}')
            write_result.append(str(np.around(r.mean().item(),2))+'+'+str(np.around(r.std().item(),2)))
            r = best_result[:, 6]
            print(f'Mean adjscore: {r.mean():.2f} ± {r.std():.2f}')
            write_result.append(str(np.around(r.mean().item(),2))+'+'+str(np.around(r.std().item(),2)))

            return write_result