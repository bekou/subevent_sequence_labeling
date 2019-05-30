from prettytable import PrettyTable
import utils

class printClasses():
    def __init__(self):
        self.t = PrettyTable(['Class', 'TP', 'FP', 'FN', 'Pr', 'Re', 'F1'])

    def add(self, Class, TP, FP, FN, Pr, Re, F1):
        self.t.add_row([Class, TP, FP, FN, Pr, Re, F1])

    def print(self):
        print(self.t)


class nerTokenEvaluator:

    def __init__(self, ner_classes=[]):

        self.NERclasses = ner_classes

        # print (self.classes)
        # print ( utils.getClassesSetERR())
        self.totals = 0
        self.oks = 0

        self.tps = 0
        self.fps = 0
        self.fns = 0

        self.tpsClasses = dict.fromkeys(self.NERclasses, 0)
        self.fpsClasses = dict.fromkeys(self.NERclasses, 0)
        self.fnsClasses = dict.fromkeys(self.NERclasses, 0)
        self.precision = dict.fromkeys(self.NERclasses, 0)
        self.recall = dict.fromkeys(self.NERclasses, 0)
        self.f1 = dict.fromkeys(self.NERclasses, 0)

    def add(self, pred, true):
        # print("add0")
        # pred_batches = [[14, 6, 14, 8, 9, 14, 14, 14, 2, 2, 14, 2, 14]]
        # true_batches = [[14, 6, 4, 8, 5, 14, 4, 2, 2, 1, 14, 0, 14]]

        # for batch_idx in range(len(pred_batches)):
        #    pred = pred_batches[batch_idx]
        #    true = true_batches[batch_idx]

        for idx in range(len(pred)):
            if pred[idx] == true[idx]:

                tlabel_name = self.NERclasses[true[idx]]
                self.tpsClasses[tlabel_name] += 1
            else:
                # print (true[idx])
                # print (pred[idx])
                tlabel_name = self.NERclasses[true[idx]]
                plabel_name = self.NERclasses[pred[idx]]

                self.fnsClasses[tlabel_name] += 1
                self.fpsClasses[plabel_name] += 1

    def getPrecision(self, tps, fps):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fps)

    def getRecall(self, tps, fns):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fns)

    def getF1(self, tps, fps, fns):
        if tps == 0:
            return 0
        else:
            return 2 * self.getPrecision(tps, fps) * self.getRecall(tps, fns) / (
                        self.getPrecision(tps, fps) + self.getRecall(tps, fns))

    def getOverallF1(self):
        self.tps = 0
        self.fns = 0
        self.fps = 0
        for label in self.NERclasses:
            # print (label)
            if label != "O":
                self.tps += self.tpsClasses[label]
                self.fns += self.fnsClasses[label]
                self.fps += self.fpsClasses[label]

        return self.getF1(self.tps, self.fps, self.fns)

    def getAccuracy(self):
        return self.oks / self.totals

    def printInfo(self):

        printer = printClasses()

        self.tps = 0
        self.fns = 0
        self.fps = 0

        for label in self.NERclasses:

            if label != "O":
                self.tps += self.tpsClasses[label]
                self.fns += self.fnsClasses[label]
                self.fps += self.fpsClasses[label]

            printer.add(label, self.tpsClasses[label], self.fpsClasses[label], self.fnsClasses[label],
                        self.getPrecision(self.tpsClasses[label], self.fpsClasses[label]),
                        self.getRecall(self.tpsClasses[label], self.fnsClasses[label]),
                        self.getF1(self.tpsClasses[label], self.fpsClasses[label], self.fnsClasses[label]))

            # print('%s TP: %d  FP: %d  FN: %d TN: %d precision: %f recall: %f F1: %f' % (label,self.tpsClasses[label],self.fpsClasses[label],self.fnsClasses[label],self.tnsClasses[label], self.precision[label], self.recall[label], self.f1[label]))
        printer.add("-", "-", "-", "-",
                    "-", "-",
                    "-")
        printer.add("Micro NER", self.tps, self.fps, self.fns,
                    self.getPrecision(self.tps, self.fps), self.getRecall(self.tps, self.fns),
                    self.getF1(self.tps, self.fps, self.fns))

        printer.print()

def getMaxOccurence(lst):
    from collections import Counter
    most_common, num_most_common = Counter(lst).most_common(1)[0]  # 4, 6 times
    return most_common

def classesToChunks(tokenClasses, chunks):
    labeled_chunks = []
    for chunk in chunks:

        class_list = (tokenClasses[chunk[1]:chunk[2] + 1])

        if chunk[0] in class_list:
            labeled_chunks.append((chunk[0], chunk[1], chunk[2]))
        else:
            labeled_chunks.append((getMaxOccurence(class_list), chunk[1], chunk[2]))
            # print (class_list)
    return labeled_chunks


def get_chunk_type(tok, idx_to_tag):
    # method implemented in https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/data_utils.py
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type
def get_chunks(seq, tags):
    # method implemented in https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/data_utils.py
    """Given a sequence of tags, group entities and their position
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """

    default = tags['O']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i-1)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i-1)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq)-1)
        chunks.append(chunk)

    return chunks

class relaxedChunkEvaluator:
    def __init__(self, tag_to_ix):
        self.tag_to_ix = tag_to_ix

        self.nerSegmentationTags, self.ECset = utils.getSortedTagsFromBIO(tag_to_ix)
        self.tag_to_ix = utils.getSegmentationDict(self.nerSegmentationTags)

        self.totals = 0
        self.oks = 0

        self.tpsNER = 0
        self.fpsNER = 0
        self.fnsNER = 0

        self.tpsNERMicro_no_other = 0
        self.fpsNERMicro_no_other = 0
        self.fnsNERMicro_no_other = 0

        self.tpsClassesNER = dict.fromkeys(self.ECset, 0)
        self.fpsClassesNER = dict.fromkeys(self.ECset, 0)
        self.fnsClassesNER = dict.fromkeys(self.ECset, 0)
        self.precisionNER = dict.fromkeys(self.ECset, 0)
        self.recallNER = dict.fromkeys(self.ECset, 0)
        self.f1NER = dict.fromkeys(self.ECset, 0)

        self.correct_predsNER, self.total_correctNER, self.total_predsNER = 0., 0., 0.

    def add(self, pred, true, trueBIO):

        #print (true)

        true=true.data.tolist()
        pred = pred.data.tolist()
        trueBIO= trueBIO#.data.tolist()

        #print (self.tag_to_ix)
        lab_chunks_ = set(get_chunks(trueBIO, self.tag_to_ix))

        #print (lab_chunks_)


        # lab_pred_chunks = set(get_chunks(predNER, tagsNER))

        lab_chunks_list_ = list(lab_chunks_)

        # print (lab_chunks_)
        #print ("self.nerSegmentationTags")
        #print (self.nerSegmentationTags)
        if true == trueBIO:
            trueNER_tags = utils.listOfIdsToTags(true, self.nerSegmentationTags)
            predNER_tags = utils.listOfIdsToTags(pred, self.nerSegmentationTags)

            #print (trueNER_tags)
            #print (predNER_tags)

            #trueEC_tags = trueNER_tags
            #predEC_tags = predNER_tags


            trueEC_tags = utils.transformToECtags(trueNER_tags)
            predEC_tags = utils.transformToECtags(predNER_tags)
        else:
            trueEC_tags = utils.listOfIdsToTags(true, self.ECset)
            predEC_tags = utils.listOfIdsToTags(pred, self.ECset)

        lab_chunks = set(classesToChunks(trueEC_tags, lab_chunks_list_))
        lab_pred_chunks = set(classesToChunks(predEC_tags, lab_chunks_list_))

        lab_chunks_list = list(lab_chunks)
        lab_pred_chunks_list = list(lab_pred_chunks)

        for lab_idx in range(len(lab_pred_chunks_list)):

            if lab_pred_chunks_list[lab_idx] in lab_chunks_list:
                # print (lab_pred_chunks_list[lab_idx][0])
                self.tpsClassesNER[lab_pred_chunks_list[lab_idx][0]] += 1
            else:
                self.fpsClassesNER[lab_pred_chunks_list[lab_idx][0]] += 1
                # fnsEntitiesNER+=1

        for lab_idx in range(len(lab_chunks_list)):

            if lab_chunks_list[lab_idx] not in lab_pred_chunks_list:
                self.fnsClassesNER[lab_chunks_list[lab_idx][0]] += 1

        self.correct_predsNER += len(lab_chunks & lab_pred_chunks)
        self.total_predsNER += len(lab_pred_chunks)
        self.total_correctNER += len(lab_chunks)

    def getPrecision(self, tps, fps):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fps)

    def getRecall(self, tps, fns):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fns)

    def getF1(self, tps, fps, fns):
        if tps == 0:
            return 0
        else:
            return 2 * self.getPrecision(tps, fps) * self.getRecall(tps, fns) / (
                    self.getPrecision(tps, fps) + self.getRecall(tps, fns))


    def getOverallF1(self):
        tps = 0
        fns = 0
        fps = 0
        tpsNERMicro_no_other=0
        fpsNERMicro_no_other = 0
        fnsNERMicro_no_other = 0

        for label in self.ECset:
            # print (label)
            #if label != "O":
            tps += self.tpsClassesNER[label]
            fns += self.fnsClassesNER[label]
            fps += self.fpsClassesNER[label]

            if "other" != label.lower():
                tpsNERMicro_no_other += self.tpsClassesNER[label]
                fpsNERMicro_no_other += self.fnsClassesNER[label]
                fnsNERMicro_no_other += self.fpsClassesNER[label]


        return self.getF1(tps, fps, fns),self.getF1(tpsNERMicro_no_other, fpsNERMicro_no_other, fnsNERMicro_no_other)

    def printInfoMicro(self):

        printer = printClasses()

        class_other_exists = False

        for label in self.ECset:
            # if label != "O" :
            self.tpsNER += self.tpsClassesNER[label]

            self.fnsNER += self.fnsClassesNER[label]
            self.fpsNER += self.fpsClassesNER[label]

            if "other" != label.lower():

                self.tpsNERMicro_no_other += self.tpsClassesNER[label]
                self.fpsNERMicro_no_other += self.fnsClassesNER[label]
                self.fnsNERMicro_no_other += self.fpsClassesNER[label]
            else:
                class_other_exists = True

            printer.add(label, self.tpsClassesNER[label], self.fpsClassesNER[label], self.fnsClassesNER[label],
                        self.getPrecision(self.tpsClassesNER[label], self.fpsClassesNER[label]),
                        self.getRecall(self.tpsClassesNER[label], self.fnsClassesNER[label]),
                        self.getF1(self.tpsClassesNER[label], self.fpsClassesNER[label], self.fnsClassesNER[label]))

            # print('%s TP: %d  FP: %d  FN: %d TN: %d precision: %f recall: %f F1: %f' % (label,self.tpsClasses[label],self.fpsClasses[label],self.fnsClasses[label],self.tnsClasses[label], self.precision[label], self.recall[label], self.f1[label]))
        printer.add("-", "-", "-", "-",
                    "-", "-",
                    "-")
        printer.add("Micro NER chunk RELAXED", self.tpsNER, self.fpsNER, self.fnsNER,
                    self.getPrecision(self.tpsNER, self.fpsNER), self.getRecall(self.tpsNER, self.fnsNER),
                    self.getF1(self.tpsNER, self.fpsNER, self.fnsNER))

        if class_other_exists == True:
            printer.add("Micro NER chunk RELAXED ^Other", self.tpsNERMicro_no_other, self.fpsNERMicro_no_other,
                        self.fnsNERMicro_no_other,
                        self.getPrecision(self.tpsNERMicro_no_other, self.fpsNERMicro_no_other),
                        self.getRecall(self.tpsNERMicro_no_other, self.fnsNERMicro_no_other),
                        self.getF1(self.tpsNERMicro_no_other, self.fpsNERMicro_no_other, self.fnsNERMicro_no_other))

        printer.print()
def getPrecision(tps, fps):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fps)

def getRecall(tps, fns):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fns)

def getF1( tps, fps, fns):
        if tps == 0:
            return 0
        else:
            return 2 * getPrecision(tps, fps) * getRecall(tps, fns) / (
                    getPrecision(tps, fps) +getRecall(tps, fns))
class relaxedEventEvaluator:

    def __init__(self, event_classes=[]):

        self.eventClasses = event_classes

        # print (self.classes)
        # print ( utils.getClassesSetERR())
        self.totals = 0
        self.oks = 0

        self.tps = 0
        self.fps = 0
        self.fns = 0

        #print (self.eventClasses)

        self.tpsClasses = dict.fromkeys(self.eventClasses, 0)
        self.fpsClasses = dict.fromkeys(self.eventClasses, 0)
        self.fnsClasses = dict.fromkeys(self.eventClasses, 0)
        self.precision = dict.fromkeys(self.eventClasses, 0)
        self.recall = dict.fromkeys(self.eventClasses, 0)
        self.f1 = dict.fromkeys(self.eventClasses, 0)

        self.total_event_types = {}
        """The types of events (e.g. goal, yellow card) for all evaluated files"""
        self.detected_events_types = {}
        """Events that were detected for the current file"""
        self.total_confusion_matrix = [0, 0, 0, 0]  # TP, TN, FP, FN
        """Confusion matrix for all evaluated files"""
        self.macro_measures = [float(0), float(0), float(0), float(0),float(0)]  # accuracy, precision, recall, #evaluated files



    def add(self, pred, true, ec_independent_targets,ec_independent_event_ids):
        # print("add0")
        # pred_batches = [[14, 6, 14, 8, 9, 14, 14, 14, 2, 2, 14, 2, 14]]
        # true_batches = [[14, 6, 4, 8, 5, 14, 4, 2, 2, 1, 14, 0, 14]]

        # for batch_idx in range(len(pred_batches)):
        #    pred = pred_batches[batch_idx]
        #    true = true_batches[batch_idx]


        pred=pred.data.tolist()
        true =true.data.tolist()
        #ec_independent_targets =ec_independent_targets.data.tolist()
        #ec_independent_event_ids =ec_independent_event_ids.data.tolist()

        #print(pred)
        #print(true)

        #ec_independent_targets_l=[]
        #for ec in ec_independent_targets:
        #    ec_independent_targets_l.append(ec)

        ec_independent_event_ids_l = []
        for ec in ec_independent_event_ids:
            ec_independent_event_ids_l.append(ec.item())

        #print(ec_independent_targets)
        #print(ec_independent_targets_l)
        #print(ec_independent_event_ids_l)



        false_positives, true_negatives = 0, 0
        unique_events_detected = set()

        events_=set(ec_independent_event_ids_l)
        #print ("set")
        #print (events_)
        if (-1 in events_) and (-2 in events_):
            len_events=len(events_)-2
        elif (-1 in events_):
            len_events=len(events_)-1
        elif (-2 in events_):
            len_events=len(events_)-1
        else:
            len_events = len(events_)


        #descriptions = {}
        #summary = []
        #summary_index = []
        #evaluation_results_file = evaluation_path + "/" + file_name[:-4] + "_eval.txt"
        #events, excluded = Evaluation.read_events(ground_truth_path + "/" + file_name)
        # print ("THE EVENTS and EXCLUDED")
        # print (events)
        # print (excluded)
        # print ("end THE EVENTS and EXCLUDED")
        #f = open(evaluation_results_file, 'w')
        period_counter = -1
        for idx in range(len(pred)):
            period_counter += 1
            #timestamps, prediction, description = period
            prediction=pred[idx]
            #gold = true[idx]

            event_id =ec_independent_event_ids [idx].item()
            event_type=ec_independent_targets[idx]
            #,  =#Evaluation.is_event(events, timestamps)

            #print ("Event type: " + event_type + ", Predicted: " + str(prediction))

            #f.write("Period [" + ", ".join(timestamps) + "], Event type: " + event_type + ", Predicted: " + str(
            #    prediction) + " \"" + description + "\"" + "\n")
            prediction_label=self.eventClasses[prediction]
            #gold_label = self.eventClasses[gold]

            #print (event_id)
            if event_id == -2:
                continue


            if prediction_label=="event" and event_type != "none" and event_id not in unique_events_detected:
                self.detected_events_types[event_type] = self.detected_events_types.get(event_type, 0) + 1

            if event_id != -1:
                if prediction_label=="event":
                    unique_events_detected.add(event_id)


            else:
                if prediction_label=="event":
                    false_positives += 1

                    self.fpsClasses[prediction_label]+= 1

                else:
                    true_negatives += 1
                    #self.tnsClasses[prediction_label] += 1


        self.tpsClasses["event"] += len(unique_events_detected)
        true_positives = len(unique_events_detected)

        #self.tpsClasses["event"] = len(unique_events_detected)

        false_negatives = len_events - len(unique_events_detected)
        self.fnsClasses["non-event"] += len_events - len(unique_events_detected)


        events_number = str(len_events)
        #if len(unique_events_detected) != 0:

        #      print ("Unique events found: " + str(len(unique_events_detected)) + ", Total events: " + events_number)

        #else:
        #    toWrite += ["0 unique events found,  Total Events: " + events_number]

        confusion_matrix = [true_positives, true_negatives, false_positives, false_negatives]
        for i in range(len(self.total_confusion_matrix)):
              self.total_confusion_matrix[i] += confusion_matrix[i]

        accuracy = (true_positives + true_negatives) / float(sum(confusion_matrix))
        precision = true_positives / float(true_positives + false_positives) if (
                                                                                              true_positives + false_positives) != 0 else 0
        recall = true_positives / float(true_positives + false_negatives) if (
                                                                                           true_positives + false_negatives) != 0 else 0
        self.macro_measures[0] += accuracy
        self.macro_measures[1] += precision
        self.macro_measures[2] += recall
        self.macro_measures[3] += self.getF1(true_positives,false_positives,false_negatives)
        self.macro_measures[4] += 1

        #print ("True positives: " + str(true_positives) + ", True negatives: " + str(true_negatives)
        #              + ", False positives: " + str(false_positives) + ", False negatives: " + str(false_negatives))
        #print ("TP")
        #print (self.tpsClasses)
        #print ("FP")
        #print (self.fpsClasses)
        # print ("FN")
        #print (self.fnsClasses)

        #print ("Precision: " + str(precision))
        #print("Recall: " + str(recall))



          #toWrite += ["True positives: " + str(true_positives) + ", True negatives: " + str(true_negatives)
          #            + ", False positives: " + str(false_positives) + ", False negatives: " + str(false_negatives)]
          #toWrite += ["Precision: " + str(precision)]
          #toWrite += ["Recall: " + str(recall)]
          #f.write(Util.beauty_print(toWrite))
          #f.close()

          #summary_split = evaluation_results_file.split("/")
          #print(summary_split)
          #summary_file = '/'.join(summary_split[:-1]) + "/" + summary_split[-1]
          #print(summary_file)
    def getPrecision(self, tps, fps):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fps)

    def getRecall(self, tps, fns):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fns)

    def getF1(self, tps, fps, fns):
        if tps == 0:
            return 0
        else:
            return 2 * self.getPrecision(tps, fps) * self.getRecall(tps, fns) / (
                    self.getPrecision(tps, fps) + self.getRecall(tps, fns))
                    
    def getMacroF1(self):
        return self.macro_measures[3]/self.macro_measures[4]
        
    def getMicroF1(self):
        self.tps = 0
        self.fns = 0
        self.fps = 0

        for label in self.eventClasses:

            #if label != "O":
            self.tps += self.tpsClasses[label]
            self.fns += self.fnsClasses[label]
            self.fps += self.fpsClasses[label]

           
        return self.getF1(self.tps, self.fps, self.fns)
    

    def getContincencyTable(self):
        self.tps = 0
        self.fns = 0
        self.fps = 0

        for label in self.eventClasses:

            #if label != "O":
            self.tps += self.tpsClasses[label]
            self.fns += self.fnsClasses[label]
            self.fps += self.fpsClasses[label]
            
            
        return self.tps,self.fns,self.fps
        
    def printInfo(self):

        printer = printClasses()

        self.tps = 0
        self.fns = 0
        self.fps = 0

        for label in self.eventClasses:

            #if label != "O":
            self.tps += self.tpsClasses[label]
            self.fns += self.fnsClasses[label]
            self.fps += self.fpsClasses[label]

            printer.add(label, self.tpsClasses[label], self.fpsClasses[label], self.fnsClasses[label],
                        self.getPrecision(self.tpsClasses[label], self.fpsClasses[label]),
                        self.getRecall(self.tpsClasses[label], self.fnsClasses[label]),
                        self.getF1(self.tpsClasses[label], self.fpsClasses[label], self.fnsClasses[label]))

            # print('%s TP: %d  FP: %d  FN: %d TN: %d precision: %f recall: %f F1: %f' % (label,self.tpsClasses[label],self.fpsClasses[label],self.fnsClasses[label],self.tnsClasses[label], self.precision[label], self.recall[label], self.f1[label]))
        printer.add("-", "-", "-", "-",
                    "-", "-",
                    "-")
        printer.add("Micro", self.tps, self.fps, self.fns,
                    self.getPrecision(self.tps, self.fps), self.getRecall(self.tps, self.fns),
                    self.getF1(self.tps, self.fps, self.fns))

        printer.add("Macro", "-", "-","-",
                    self.macro_measures[1]/self.macro_measures[4], self.macro_measures[2]/self.macro_measures[4],
                    self.macro_measures[3]/self.macro_measures[4])

        printer.print()