import random

from nltk import word_tokenize
from collections import Counter
from operator import itemgetter


def dot(dictA, dictB):
    # listA = list(dictA.values())  # Lukas: Transformation in Liste wird nicht benötigt
    # listB = list(dictB.values())
    # dotproduct = sum([x * y for (x,y) in zip(listA, listB)])
    # dotproduct = sum(map(lambda pair:pair[0]*pair[1], zip(listA,listB)))
    # dotproduct = np.dot(listA,listB)
    dotproduct = sum(dictA[key]*dictB.get(key, 0) for key in dictA)
    return dotproduct
    # TODO: Ex. 2: return vector product between features vectors represented by dictA and dictB.
    # Ich habe die builtin zip function verwendet, und dann jeweils die die Vektoren multipliziert.
    # Vielleicht sollten wir noch etwas einbauen um mit Vektoren umzugehen, die nicht gleich lang sind?
    # test return: "FAIL", die Summe stimmt nicht # Lukas: Behoben
    # Lukas: Summe stimmt nicht, weil die funktion nur identische werte aus den dictionairies aufsummieren soll (also car*car + apple*apple). Hier summiert sie aber banana und house auf, obwohl beide in ihrem Vektor jeweils 0 sind.


def normalized_tokens(text):
    return [token.lower() for token in word_tokenize(text)]

class DataInstance:
    def __init__(self, feature_counts, label):
        """ A data instance consists of a dictionary with feature counts (string -> int) and a label (True or False)."""
        self.feature_counts = feature_counts     #
        self.label = label

    @classmethod
    def from_list_of_feature_occurrences(cls, feature_list, label):
        """ Creates feature counts for all features in the list."""
        feature_counts = dict()
        # TODO: Ex. 3: create a dictionary that contains for each feature in the list the count how often it occurs.
        # PSEUDOCODE: vergleiche feature_counts mit feature_list
        # if item not in feature_counts add item with value "1" to dic
        # if item in feature_counts add +1 to value
        # test returned "ok"

        for i in feature_list:
            if i not in feature_counts:
                feature_counts[i] = 1
            else:
                feature_counts[i] = feature_counts[i] + 1
        return cls(feature_counts, label)

        # Lukas: vielleicht feature_list als set anlegen? for i in set(feature_list), um Iterationen zu reduzieren?

    @classmethod
    def from_text_file(cls, filename, label):
        with open(filename, 'r') as myfile:
            token_list = normalized_tokens(myfile.read().strip())
        return cls.from_list_of_feature_occurrences(token_list, label)

class Dataset:
    instance_list: object

    def __init__(self, instance_list):
        """ A data set is defined by a list of instances """
        self.instance_list = instance_list      # Lukas: Attribut der Klasse Datensatz ist instanz list, eine liste mit Instanzen. Die Instanzen werden definiert durch die Klasse DataInstace (deren Atribute:
        # feature_counts (dictionairy) und label (true/false)
        self.feature_set = set.union(*[set(inst.feature_counts.keys()) for inst in instance_list])  # Lukas: Hier wird feature_set aus allen Instanzen erzeugt, die angegeben werden


    def get_topn_features(self, n):
        """ This returns a set with the n most frequently occurring features (i.e. the features that are contained in most instances)."""
        # Lukas: Soll die n häufigsten Merkmale finden.
        # Lukas: Ableitbar aus Dokument-frequency: Wie n-oft kommt Wert in Dokument x vor?
        # Lukas: Lösung mit most_common (funktioniert):
        init = Counter()

        for instance in self.instance_list:
           init.update([feature for feature in instance.feature_counts.keys()])
        return set(feature for feature, count in init.most_common(n)) # TODO: Ex. 4: Return set of features that occur in most instances.  # Funktioniert

        #top_n = sorted(self.instance_list, key=i, reverse=False)[:n]   # Funktioniert

        #return set(top_n)  # TODO: Ex. 4: Return set of features that occur in most instances.  #Lukas: Fehlerbehebung funktioniert nicht
        # Lukas: Ältere Versuche:

        # Lukas: funktioniert nicht

        #top_n = sorted(self.instance_list, key= lambda i, reverse = True)[:n]  # Lukas: Syntax error

       # for inst in self.instance_list:
          #  top_n = sorted(self.instance_list, key= inst.feature_counts.keys(), reverse= True)[:n]
        # top_n = sorted(self.instance_list, key= itemgetter(1), reverse=True)[:n]

    def set_feature_set(self, feature_set):
        """
        This restrics the feature set. Only features in the specified set all retained. All other feature are removed
        from all instances in the dataset AND from the feature set."""
        # TODO: Ex. 5: Filter features according to feature set.
        # Lukas: Soll, glaube ich, auf eine Merkmalsmenge an features weiter einschränken und ein Subset erzeugen, da z.B. seltene Merkmale Ausreißer erzeugen
        # Lukas: z.B. um Overfitting zu vermeiden.
        # Lukas: Pseudocode: instance_list dictionairy(feature_counts) sollen kleine Werte rausgeschmissen werden
        # Lukas: feature_counts_old = {cat: 10, hill: 12, men: 4, shoe:1} --> feature_counts_new = {cat: 10, hill: 12, men: 4} --> shoe fliegt raus durch filtern mittels feature_set
        ### Sarah: also dataset mit feature_set vergleichen? ' Lukas: Hier sollen, glaube ich, top_n features mit dataset verglichen werden und sämtliche aus feature_set gelöscht werden, die nicht vorkommen

        self.feature_set = feature_set

        for inst in self.instance_list:
            copy_feature_count = dict(inst.feature_counts)
            for feature in set(copy_feature_count) - feature_set:
                del inst.feature_counts[feature]

        #pass


    def most_frequent_sense_accuracy(self):
        """ Computes the accuracy of always predicting the overall most frequent sense for all instances in the dataset. """

        # Sarah: Idee, wenn angenommen wird, dass alles als most frequent label klassifiziert wird
        # dann müsst die accuracy das most_frquent_label/jedes item im data set sein.
        #most_frequent_label = max(set(self.label), key= self.label.count)
        #return most_frequent_label/self.count
        # sagt für jede instanz im datensatz die häufigste Kategorie vor

        true_weight = 0
        false_weight = 0

        for instance in self.instance_list:
            if instance.label == True:
                true_weight += 1
        else:
            false_weight += 1

        frequency_label_weight = true_weight+false_weight
        return frequency_label_weight / len(self.instance_list)

        #return 0.0 # TODO: Ex. 6: Return accuracy of always predicting most frequent label in data set.

    def shuffle(self):
        """ Shuffles the dataset. Beneficial for some learning algorithms."""
        random.shuffle(self.instance_list)
