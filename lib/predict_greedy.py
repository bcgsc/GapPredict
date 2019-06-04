from lstm.GapPredictModel import GapPredictModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceReverser import SequenceReverser
from utils.DataWriter import DataWriter
import utils.directory_utils as UTILS
from preprocess.KmerLabelDecoder import KmerLabelDecoder
from predict.GuidedPredictor import GuidedPredictor
from predict.RandomPredictor import RandomPredictor
from predict.BeamSearchPredictor import BeamSearchPredictor
from predict.TeacherForceProbabilityPredictor import TeacherForceProbabilityPredictor

def predict_arbitrary_length(weights_path, gap_id, fasta_path, embedding_dim, latent_dim, length_to_predict, base_path=None):
    importer = SequenceImporter()
    reverser = SequenceReverser()

    sequences = importer.import_fasta([fasta_path])

    left_flank = sequences[0]
    right_flank = reverser.reverse_complement(sequences[1])

    model = GapPredictModel(min_seed_length=None, stateful=False, batch_size=1, embedding_dim=embedding_dim,
                            latent_dim=latent_dim, with_gpu=True)

    model.load_weights(weights_path + 'my_model_weights.h5')

    forward_predict = predict_gaps(model, left_flank, length_to_predict, base_path=base_path, directory="predict_gap/forward")
    reverse_predict = predict_gaps(model, right_flank, length_to_predict, base_path=base_path, directory="predict_gap/reverse_complement")

    writer = DataWriter(root_directory=base_path)
    postfix = "_LD_"+str(latent_dim)
    writer.save_complements(forward_predict, reverse_predict, gap_id, postfix=postfix, fasta_ref=fasta_path)

def predict_reference(weights_path, gap_id, fasta_path, embedding_dim, latent_dim, min_seed_length, plots=False, base_path=None):
    importer = SequenceImporter()
    reverser = SequenceReverser()

    sequences = importer.import_fasta([fasta_path])[0:2]

    stateful_model = GapPredictModel(min_seed_length=min_seed_length, stateful=True, batch_size=1, embedding_dim=embedding_dim,
                            latent_dim=latent_dim, with_gpu=True)
    stateless_model = GapPredictModel(min_seed_length=min_seed_length, stateful=False, batch_size=1, embedding_dim=embedding_dim,
                            latent_dim=latent_dim, with_gpu=True)

    stateful_model.load_weights(weights_path + 'my_model_weights.h5')
    stateless_model.load_weights(weights_path + 'my_model_weights.h5')

    terminal_directory_character = UTILS.get_terminal_directory_character()
    first_directory = "regenerate_seq" + terminal_directory_character
    forward_left_flank = None
    rc_left_flank = None
    forward_right_flank = None
    rc_right_flank = None
    for i in range(len(sequences)):
        sequence = sequences[i]

        #TODO: messy since we assume that i will always be [0, 1]
        if i == 0:
            static_path = first_directory + "left_flank" + terminal_directory_character
        elif i == 1:
            static_path = first_directory + "right_flank" + terminal_directory_character

        forward_predict = predict_flanks(stateful_model, stateless_model, min_seed_length, sequence, base_path=base_path, directory=static_path + "forward")
        reverse_predict = predict_flanks(stateful_model, stateless_model, min_seed_length, reverser.reverse_complement(sequence), base_path=base_path, directory=static_path + "reverse_complement")

        if i == 0:
            forward_left_flank = forward_predict
            rc_left_flank = reverse_predict
        elif i == 1:
            forward_right_flank = forward_predict
            rc_right_flank = reverse_predict

    writer = DataWriter(root_directory=base_path, directory=first_directory)
    writer.write_flank_predict_fasta(forward_left_flank, rc_left_flank, forward_right_flank, rc_right_flank, latent_dim, gap_id)

def predict_flanks(stateful_model, stateless_model, min_seed_length, sequence, base_path=None, directory=None):
    writer = DataWriter(root_directory=base_path, directory=directory)
    guided_predictor = GuidedPredictor(stateful_model)
    greedy_predictor = BeamSearchPredictor(stateless_model)
    random_predictor = RandomPredictor(stateful_model)
    teacher_force_predictor = TeacherForceProbabilityPredictor(stateful_model)

    predicted_string_with_seed, basewise_probabilities = guided_predictor.regenerate_sequence(min_seed_length, sequence)
    teacher_force_probabilities = teacher_force_predictor.get_probabilities(min_seed_length, sequence)

    seed = sequence[:min_seed_length]
    length_to_predict = len(sequence) - min_seed_length

    predicted_string_greedy_with_seed, basewise_probabilities_greedy = greedy_predictor.predict_next_n_bases_greedy(seed, length_to_predict)

    predicted_string_random_with_seed, random_probability_vector = random_predictor.predict_random_sequence(seed, length_to_predict)

    writer.save_probabilities(basewise_probabilities)
    writer.save_probabilities(basewise_probabilities_greedy, file_id="greedy")
    writer.save_probabilities(random_probability_vector, file_id="random")
    writer.save_probabilities(teacher_force_probabilities, file_id="teacher_force")
    return predicted_string_with_seed

def predict_gaps(model, seed, prediction_length, base_path=None, directory=None):
    predictor = BeamSearchPredictor(model)
    decoder = KmerLabelDecoder()
    predicted_string_with_seed, basewise_probabilities = predictor.predict_next_n_bases_greedy(seed, prediction_length)

    writer = DataWriter(root_directory=base_path, directory=directory)
    writer.save_probabilities(basewise_probabilities)

    return decoder.decode(predicted_string_with_seed)