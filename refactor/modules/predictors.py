import torch
from typing import Iterable
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.models import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from tqdm import tqdm


class Predictor:
    def __init__(self, model: Model, iterator: DataIterator, cuda_device: int = -1):
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def predict(self, dataset: Iterable[Instance]) -> None:
        pred_generator = self.iterator(dataset, num_epochs=1, shuffle=False)
        pred_generator_tqdm = tqdm(
            pred_generator, total=self.iterator.get_num_batches(dataset)
        )

        self.model.eval()
        with torch.no_grad():
            batches_this_epoch = 0
            pred_loss = 0
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                output_dict = self.model(**batch)
                loss = output_dict["loss"]
                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    batches_this_epoch += 1
                    pred_loss += loss.detach().cpu().numpy()

                # Update the description with the latest metrics
                pred_metrics = training_util.get_metrics(
                    self.model, pred_loss, batches_this_epoch
                )
                description = training_util.description_from_metrics(pred_metrics)
                pred_generator_tqdm.set_description(description, refresh=False)

        return pred_metrics
