import numpy as np
import tensorflow as tf
import os, re, io, argparse
from jamo import hangul_to_jamo
from hparams import hparams
from librosa import effects
from models import create_model
from util.text import text_to_sequence, sequence_to_text
from util import audio, plot


sentences = [
  '흔들리는 꽃들 속에서 네 샴푸향이 느껴진거야'
]


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])
      self.alignments = self.model.alignments[0]
      self.inputs = self.model.inputs[0]

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text, base_path, idx):
    seq = text_to_sequence(text)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    input_seq, wav, alignment = self.session.run([self.inputs, self.wav_output, self.alignments], feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    wav = wav[:audio.find_endpoint(wav)]
    out = io.BytesIO()
    audio.save_wav(wav, out)
    input_seq = sequence_to_text(input_seq)
    plot.plot_alignment(alignment, '%s-%d-align.png' % (base_path, idx), input_seq)
    return out.getvalue()


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    jamo = ''.join(list(hangul_to_jamo(text)))
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(jamo, base_path, i))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  run_eval(args)


if __name__ == '__main__':
  main()
