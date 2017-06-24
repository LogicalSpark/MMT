import math
import time
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import logging
import os
import subprocess

def checkBleuTrend(bleuDev, history, threshold):

    logger = logging.getLogger('onmt.checkBleuTrend')
    logger.info('checkBleuTrend')

    historyAverage = 0.0

    actualEpoch = len(bleuDev)

    # return False if the number of epochs is not larger than history
    if actualEpoch <= history:
        return False

    for epoch in range(history):
        historyAverage += bleuDev[actualEpoch - epoch - 2]
    historyAverage /= history

    # return False if the average of the previous history values is 0.0
    if historyAverage <= 0.0:
        return False

    # return True if the last bleuDev is smaller than
    # a given threshold with respect to the average
    # of the previous history values is
    if bleuDev[actualEpoch-1] <= (1.0 + float(threshold) / 100) * historyAverage:
        return True

    return False

class Trainer(object):
    class Options(object):
        def __init__(self):
            self.save_model = None  # Set by train

            self.seed = 3435
            self.gpus = range(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            self.log_interval = 50

            # Model options --------------------------------------------------------------------------------------------
            self.encoder_type = "text"  # type fo encoder (either "text" or "img"
            self.layers = 2  # Number of layers in the LSTM encoder/decoder
            self.rnn_size = 500  # Size of LSTM hidden states
            self.word_vec_size = 500  # Word embedding sizes
            self.input_feed = 1  # Feed the context vector at each time step as additional input to the decoder
            self.brnn = True  # Use a bidirectional encoder
            self.brnn_merge = 'sum'  # Merge action for the bidirectional hidden states: [concat|sum]

            # Optimization options -------------------------------------------------------------------------------------
            self.batch_size = 64  # Maximum batch size
            self.max_generator_batches = 32  # Maximum batches of words in a seq to run the generator on in parallel.
            self.epochs = 30  # Number of training epochs
            self.start_epoch = 1  # The epoch from which to start
            self.param_init = 0.1  # Parameters are initialized over uniform distribution with support
            self.optim = 'sgd'  # Optimization method. [sgd|adagrad|adadelta|adam]
            self.max_grad_norm = 5  # If norm(gradient vector) > max_grad_norm, re-normalize
            self.dropout = 0.3  # Dropout probability; applied between LSTM stacks.
            self.curriculum = False
            self.extra_shuffle = False  # Shuffle and re-assign mini-batches

            # Learning rate --------------------------------------------------------------------------------------------
            self.learning_rate = 1.0
            self.learning_rate_decay = 0.9
            self.start_decay_at = 10

            # Pre-trained word vectors ---------------------------------------------------------------------------------
            self.pre_word_vecs_enc = None
            self.pre_word_vecs_dec = None

        def state_dict(self):
            return self.__dict__

        def load_state_dict(self, d):
            self.__dict__ = d
            # we force the encoder type to "text";
            # this trick makes the models build with an old version of the software compatible with the new version
            self.encoder_type = "text"  # type fo encoder (either "text" or "img"

        def __repr__(self):
            return repr(self.__dict__)

    def __init__(self, opt):
        self.opt = opt

        self.terminate = False  #termination flag
        self.bleuDev = []
        self.minimumBleuIncrement = 10
        self.minimumEpochs = 3
        self.bleuScore = "mmt-bleu.perl"

        self._logger = logging.getLogger('onmt.Trainer')
        self._logger.info('Training Options:%s' % self.opt)

    def NMTCriterion(self, vocabSize):
        opt = self.opt
        weight = torch.ones(vocabSize)
        weight[onmt.Constants.PAD] = 0
        crit = nn.NLLLoss(weight, size_average=False)
        if opt.gpus:
            crit.cuda()
        return crit

    def memoryEfficientLoss(self, outputs, targets, generator, crit, eval=False):
<<<<<<< HEAD
        opt=self.opt
=======
        opt = self.opt
>>>>>>> features/neural_decoder
        # compute generations one piece at a time
        num_correct, loss = 0, 0
        outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

        sentence_size = outputs.size(0)
        batch_size = outputs.size(1)

        outputs_split = torch.split(outputs, opt.max_generator_batches)
        targets_split = torch.split(targets, opt.max_generator_batches)

        validPredictions = torch.IntTensor(sentence_size,batch_size)

        for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
            out_t = out_t.view(-1, out_t.size(2))
            scores_t = generator(out_t)
            loss_t = crit(scores_t, targ_t.view(-1))
            pred_t = scores_t.max(1)[1]

            # self._logger.info("def Trainer::memoryEfficientLoss i:%d repr(targ_t.data.tolist()):%s" % (i, repr(targ_t.data.tolist())))


            for h in range(sentence_size):
                for k in range(batch_size):
                    idx = h * batch_size + k
                    validPredictions[h][k] = pred_t.data[idx][0]

            # self._logger.info("def Trainer::memoryEfficientLoss validPredictions.tolist():%s" % (repr(validPredictions.tolist())))

            num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(onmt.Constants.PAD).data).sum()
            num_correct += num_correct_t
            loss += loss_t.data[0]
            if not eval:
                loss_t.div(batch_size).backward()

        grad_output = None if outputs.grad is None else outputs.grad.data
        return loss, grad_output, num_correct, validPredictions.transpose(0,1)

    def eval(self, model, criterion, data):
        total_loss = 0
        total_words = 0
        total_num_correct = 0

        model.eval()
        for i in range(len(data)):
            batch = data[i][:-1] # exclude original indices

            outputs = model(batch)
            targets = batch[1][1:]  # exclude <s> from targets
            loss, _, num_correct, hypotheses = self.memoryEfficientLoss(outputs, targets, model.generator, criterion, eval=True)
            references = targets.data.transpose(0,1)
            total_loss += loss
            total_num_correct += num_correct
            total_words += targets.data.ne(onmt.Constants.PAD).sum()

        model.train()
        return total_loss / total_words, float(total_num_correct) / total_words, hypotheses, references

    def trainModel(self, model_ori, trainData, validData, dataset, optim_ori, working_dir="/tmp", save_all_epochs=True, save_last_epoch=False, epochs=None, clone=False):


        opt = self.opt

        #At the beginning of the training, set the termination flag to False
        self.terminate = False

        if epochs:
            opt.epochs = epochs

        model = model_ori
        optim = optim_ori

        model.decoder.attn.applyMask(None)  # set the mask to None; required when the same model is trained after a translation

        model.train()

        save_last_epoch = save_last_epoch and not save_all_epochs

        # define criterion of each GPU
        criterion = self.NMTCriterion(dataset['dicts']['tgt'].size())

        start_time = time.time()

        def trainEpoch(epoch):
            if opt.extra_shuffle and epoch > opt.curriculum:
                trainData.shuffle()

            # shuffle mini batch order
            batchOrder = torch.randperm(len(trainData))

            total_loss, total_words, total_num_correct = 0, 0, 0
            report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
            start = time.time()
            for i in range(len(trainData)):

                batchIdx = batchOrder[i] if epoch > opt.curriculum else i
                batch = trainData[batchIdx][:-1] # exclude original indices

                model.zero_grad()
                outputs = model(batch)
                targets = batch[1][1:]  # exclude <s> from targets
                loss, gradOutput, num_correct, _ = self.memoryEfficientLoss(
                    outputs, targets, model.generator, criterion)

                outputs.backward(gradOutput)

                # update the parameters
                optim.step()

                num_words = targets.data.ne(onmt.Constants.PAD).sum()

                report_loss += loss
                report_num_correct += num_correct
                report_tgt_words += num_words
                report_src_words += batch[0][1].data.sum()
                total_loss += loss
                total_num_correct += num_correct
                total_words += num_words
                if i % opt.log_interval == -1 % opt.log_interval:
                    self._logger.info(
                        "trainEpoch epoch %2d, %5d/%5d; num_corr: %6.2f; %3.0f src tok; %3.0f tgt tok; acc: %6.2f; ppl: %6.2f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                        (epoch, i + 1, len(trainData),
                         report_num_correct,
                         report_src_words,
                         report_tgt_words,
                         (float(report_num_correct) / report_tgt_words) * 100,
                         math.exp(report_loss / report_tgt_words),
                         report_src_words / (time.time() - start),
                         report_tgt_words / (time.time() - start),
                         time.time() - start_time))

                    report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                    start = time.time()

            return total_loss / total_words, float(total_num_correct) / total_words

        valid_acc, valid_ppl = None, None
        for epoch in range(opt.start_epoch, opt.epochs + 1):

            self._logger.info('Training epoch %g... START' % epoch)
            start_time_epoch = time.time()

            #  (1) train for one epoch on the training set
            train_loss, train_acc = trainEpoch(epoch)
            train_ppl = math.exp(min(train_loss, 100))
            self._logger.info('trainEpoch Epoch %g Train loss: %g perplexity: %g accuracy: %g' % (
                epoch, train_loss, train_ppl, (float(train_acc) * 100)))

            if validData:
                #  (2) evaluate on the validation set
                valid_loss, valid_acc, hypotheses, references = self.eval(model, criterion, validData)

                # self._logger.info('def Trainer::trainEpoch hypotheses.tolist():%s' % repr(hypotheses.tolist()))
                # self._logger.info('def Trainer::trainEpoch references.tolist():%s' % repr(references.tolist()))


                valid_ppl = math.exp(min(valid_loss, 100))
                self._logger.info('trainModel Epoch %g Validation loss: %g perplexity: %g accuracy: %g' % (
                    epoch, valid_loss, valid_ppl, (float(valid_acc) * 100)))

                #  (3) update the learning rate
                optim.updateLearningRate(valid_loss, epoch)

                self._logger.info("trainModel Epoch %g Decaying learning rate to %g" % (epoch, optim.lr))

            if save_all_epochs or save_last_epoch:
                model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
                model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
                generator_state_dict = model.generator.module.state_dict() if len(
                    opt.gpus) > 1 else model.generator.state_dict()
                opt_state_dict = opt.state_dict()

                #  (4) drop a checkpoint
                checkpoint = {
                    'model': model_state_dict,
                    'generator': generator_state_dict,
                    'dicts': dataset['dicts'],
                    'opt': opt_state_dict,
                    'epoch': epoch,
                    'optim': optim
                }


                if valid_acc is not None:
                    torch.save(checkpoint,
                               '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, 100 * valid_acc, valid_ppl, epoch))
                else:
                    torch.save(checkpoint, '%s_acc_NA_ppl_NA_e%d.pt' % (opt.save_model, epoch))

            self._logger.info('Training epoch %g... END %.2fs' % (epoch, time.time() - start_time_epoch))

            if validData:
                # if the development setvalidation Set is provided
                # compute bleuScore on the validation set for the last epoch

                # write hypotheses and references to files

                hypFilepath = os.path.join(working_dir, 'hyp_epoch' + str(epoch))
                refFilepath = os.path.join(working_dir, 'ref_epoch' + str(epoch))
                bleuFilepath = os.path.join(working_dir, 'bleu_epoch' + str(epoch))

                hypF = open(hypFilepath, "w")
                refF = open(refFilepath, "w")


                self._logger.info('Computing BLEU epoch %g... START' % epoch)
                start_time2_epoch = time.time()
                for i in range(len(validData)):

                    hypCodes = [x.tolist() for x in hypotheses]
                    refCodes = [x.tolist() for x in references]
                    for j in range(len(hypCodes)):
                        # codes = [str(x) for x in hypCodes[j]]
                        codes = [str(x) for x in hypCodes[j] if x != onmt.Constants.BOS and x != onmt.Constants.EOS and x != onmt.Constants.PAD]
                        # codes = [str(x) for x in refCodes[j] if x != onmt.Constants.BOS and x != onmt.Constants.EOS and x != onmt.Constants.PAD]
                        hypF.write(" ".join(codes)+'\n')

                    for j in range(len(refCodes)):
                        # codes = [str(x) for x in refCodes[j]]
                        codes = [str(x) for x in refCodes[j] if x != onmt.Constants.BOS and x != onmt.Constants.EOS and x != onmt.Constants.PAD]
                        refF.write(" ".join(codes)+'\n')

                hypF.close()
                refF.close()

                hypF = open(hypFilepath,'r')
                bleuF = open(bleuFilepath,'w')
                FNULL = open(os.devnull, 'w')

                # run a process to compute BLEU score
                cmd = ["perl",self.bleuScore, refFilepath]
                process = subprocess.Popen(cmd, stdin=hypF, stdout=bleuF, stderr=FNULL)
                process.wait()
                bleuF.flush()
                bleuF.write('\n')
                bleuF.close()
                self._logger.info('cmd: %s' % repr(cmd))

                bleuF = open(bleuFilepath, 'r')
                bleu = bleuF.readline().split(' ')
                self.bleuDev.append(float(bleu[0]))
                bleuF.close()

                self._logger.info('BLEU on the validation set: %s' % str(bleu[0]))

                self._logger.info('Computing BLEU epoch %g... END %.2fs' % (epoch, time.time() - start_time2_epoch))
                # check if training should continue or not
                self.terminate = checkBleuTrend(self.bleuDev, self.minimumEpochs, self.minimumBleuIncrement)

                if self.terminate == True:
                    self._logger.info('Training is ended because the termination condition has been fired; use model at epoch %d' % (epoch - 1))
                    break


        # return the previous last epoch if the termination condition has been fired; the last epoch otherwise
        if self.terminate:
            return epoch - 1
        else:
            return epoch

