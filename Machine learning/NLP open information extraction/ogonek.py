# Copyright (c) 2019, Tom SF Haines
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import csv
import zipfile

import string
import re

import numpy



"""A minimal libarary for a few NLP tasks. Created because I can't get a proper NLP library installed on university systems; also contains conveniance stuff for loading data sets."""

version = [1, 0, 4] # Interface / major / minor



###############################################################################
# Tokenisation...
###############################################################################

class Tokenise:
  """Tokenises English using a bunch of simple rules. Works well enough for well written text, but rule based so not particularly robust."""
  keepdot = {'Dr', 'Mr', 'Ms', 'Mrs', 'Miss', 'Co', 'Cie', 'St', 'Ave', 'e.g', 'i.e'}
  abbrev = {'&' : ['and'],
            "can't" : ['can', 'not'],
            "it's" : ['it', 'is'],
            "he's" : ['he', 'is'],
            "that's" : ['that', 'is'],
            "i'll" : ['i', 'will'],
            "i'd" : ['i', 'would'],
            "i'm" : ['i', 'am'],
            "i've" : ['i', 'have'],
            "they'll" : ['they', 'will'],
            "they've" : ['they', 'have'],
            "what's" : ['what', 'is'],
            "you're" : ['you', 'are'],
            "we're" : ['we', 'are'],
            "we've" : ['we', 'have'],
            "we'll" : ['we', 'will'],
            "we'd" : ['we', 'had'],
            "you'll" : ['you', 'will'],
            "let's" : ['let', 'us'],
            "sci-fi" : ['science', 'fiction']}
  
  def __init__(self, text):
    """Constructed with an arbitrary blob of text."""
    
    # Substitute UTF-8 characters that are inconveniant...
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('’', "'")
    text = text.replace('—', "-")
    
    # Split by spaces...
    toks1 = text.split()
    
    # Move punctuation from start/end of tokens into seperate tokens...
    toks2 = []
    
    for tok in toks1:
      # If parsing a number pretend a dot is not punctuation...
      if any(char.isdigit() for char in tok):
        punc = "!\"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~"
      else:
        punc = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
      
      # Remove punctuation from front of word...
      while len(tok)>0 and tok[0] in punc:
        toks2.append(tok[0])
        tok = tok[1:]
      
      # If it was only punctuation stop here!..
      if len(tok)==0:
        continue
      
      # Remove punctuation from tail of word...
      tail = []
      while tok[-1] in punc:
        if tok[-1]=='.' and (tok[:-1] in self.keepdot or (len(tok)==2 and tok[0] in string.ascii_uppercase)):
          break
        tail.append(tok[-1])
        tok = tok[:-1]
      
      # Copy over whatever is left plus the tail...
      toks2.append(tok)
      
      for t in tail[::-1]:
        toks2.append(t)
    
    # Swap out known abbreviations...
    toks3 = []
    for tok in toks2:
      if tok.lower() in self.abbrev: # Abbreviations
        for i, t in enumerate(self.abbrev[tok.lower()]):
          if i==0 and tok[0].isupper():
            toks3.append(t[0].upper() + t[1:])
            
          else:
            toks3.append(t)
      
      elif tok.endswith("n't"): # So many of these it's easier to hardcode
        toks3.append(tok[:-3])
        toks3.append('not')
        
      elif tok.endswith("'s"): # Helpful to seperate these
        toks3.append(tok[:-3])
        toks3.append("'s")
        
      else:
        toks3.append(tok)
    
    # Split words linked by dashes...
    toks4 = []
    for tok in toks3:
      if '-' in tok: # Words with a dash in them
        first = True
        for t in tok.split('-'):
          if first:
            first = False
          else:
            toks4.append('-')
          if len(t)>0:
            toks4.append(t)
      
      else:
        toks4.append(tok)
    
    # Seperate into sentences...
    self._sentence = []
    new = True
    
    for tok in toks4:
      # Keep quotes around sentences, for spoken dialog..
      if new and tok=='"' and len(self._sentence)>0 and len(self._sentence[-1])>0 and self._sentence[-1][0]=='"' and self._sentence[-1][-1]!='"':
        self._sentence[-1].append(tok) 
        continue
      
      # Create new sentence if required...
      if new:
        self._sentence.append([])
        new = False
      
      # Copy over token, unless this results in a sentence starting "", in which case donate it back to the previous sentence (dialog)...
      if tok=='"' and len(self._sentence[-1])==1 and self._sentence[-1][0]=='"' and len(self._sentence)>1:
        self._sentence[-2].append(tok)
      
      else:
        self._sentence[-1].append(tok)
      
      # Detect end of sentence...
      if tok in ['.', '!', '?', ':']:
        new = True
        
        # To handle '...'...
        if len(self._sentence[-1])==1:
          self._sentence[-1] = []
          new = False

  
  def __len__(self):
    return len(self._sentence)
  
  
  def __getitem__(self, index):
    return self._sentence[index]



###############################################################################
# Word vectors...
###############################################################################

class Glove:
  """Simple wrapper around the Glove word vectors. Be warned that it just loads everything into memory on construction - not really appropriate for the full model, but fine for a shrunk version. See https://nlp.stanford.edu/projects/glove for more information about Glove."""

  def __init__(self, fn = 'baby_glove.zip'):
    """Can optionally provide a filename to load, in Glove format (zip file, containing a text file with one line per word, as a word followed by numbers of the vector, seperated by whitespace). Defaults to a baby version I created that's small enough to distribute on Moodle."""
    
    # Dictionary from token to word vector (numpy.array)...
    self._lookup = {}
    self.length = None
    
    # Load and read in data...
    with zipfile.ZipFile(fn, 'r') as zipin:
      with zipin.open(fn[:-3] + 'txt') as fin:
        for line in fin:
          tokens = line.decode('utf8').split()
          arr = numpy.array([float(token) for token in tokens[1:]])
          arr.setflags(write=False)
          self._lookup[tokens[0]] = arr
          
          if self.length is None:
            self.length = arr.shape[0]
            
          else:
            assert(self.length==arr.shape[0])
    
    # Create zero vector for when required...
    self.zeros = numpy.zeros(self.length)
    self.zeros.setflags(write=False)

  
  def len_vec(self):
    """Returns the length of the word vectors (300 by default)."""
    return self.length
  
  
  def __len__(self):
    """Returns how many tokens it has word vectors for."""
    return len(self._lookup)
  
  
  def __contains__(self, token):
    """Lets you check if a token is within the list of word vectors supported."""
    return token.lower() in self._lookup


  def __getitem__(self, token):
    """Lets you get a token's word vector. Note that it will raise an error if it does not exist, unlike decode."""
    return self._lookup[token.lower()]


  def decode(self, token):
    """Convert a single token into a word vector. Do not edit returned word vector. Returns a vector of zeros if it doesn't know the word."""
    return self._lookup.get(token.lower(), self.zeros)
  
  
  def decodes(self, tokens):
    """Converts a list of tokens into a list of word vectors. Do not edit word vectors."""
    return [self.decode(token) for token in tokens]



###############################################################################
# Part of speech...
###############################################################################

# Description for each part of speech tag...
pos_desc = {'CC' : 'Coordinating conjunction',
            'CD' : 'Cardinal number',
            'DT' : 'Determiner',
            'EX' : 'Existential there',
            'FW' : 'Foreign word',
            'IN' : 'Preposition or subordinating conjunction',
            'JJ' : 'Adjective',
            'JJR' : 'Adjective, comparative',
            'JJS' : 'Adjective, superlative',
            'LS' : 'List item marker',
            'MD' : 'Modal',
            'NN' : 'Noun, singular or mass',
            'NNS' : 'Noun, plural',
            'NNP' : 'Proper noun, singular',
            'NNPS' : 'Proper noun, plural',
            'PDT' : 'Predeterminer',
            'POS' : 'Possessive ending',
            'PRP' : 'Personal pronoun',
            'PRP$' : 'Possessive pronoun',
            'RB' : 'Adverb',
            'RBR' : 'Adverb, comparative',
            'RBS' : 'Adverb, superlative',
            'RP' : 'Particle',
            'SYM' : 'Symbol',
            'TO' : 'to',
            'UH' : 'Interjection',
            'VB' : 'Verb, base form',
            'VBD' : 'Verb, past tense',
            'VBG' : 'Verb, gerund or present participle',
            'VBN' : 'Verb, past participle',
            'VBP' : 'Verb, non-3rd person singular present',
            'VBZ' : 'Verb, 3rd person singular present',
            'WDT' : 'Wh-determiner',
            'WP' : 'Wh-pronoun',
            'WP$' : 'Possessive wh-pronoun',
            'WRB' : 'Wh-adverb'}



# Part of speech tags to indices; useful for indexing a vector/matrix etc.
pos_to_num = {'CC' : 0,
              'CD' : 1,
              'DT' : 2,
              'EX' : 3,
              'FW' : 4,
              'IN' : 5,
              'JJ' : 6,
              'JJR' : 7,
              'JJS' : 8,
              'LS' : 9,
              'MD' : 10,
              'NN' : 11,
              'NNS' : 12,
              'NNP' : 13,
              'NNPS' : 14,
              'PDT' : 15,
              'POS' : 16,
              'PRP' : 17,
              'PRP$' : 18,
              'RB' : 19,
              'RBR' : 20,
              'RBS' : 21,
              'RP' : 22,
              'SYM' : 23,
              'TO' : 24,
              'UH' : 25,
              'VB' : 26,
              'VBD' : 27,
              'VBG' : 28,
              'VBN' : 29,
              'VBP' : 30,
              'VBZ' : 31,
              'WDT' : 32,
              'WP' : 33,
              'WP$' : 34,
              'WRB' : 35}



# Opposite of above (list so you can index by above number)...
num_to_pos = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN',
              'JJ', 'JJR', 'JJS', 'LS', 'MD',
              'NN', 'NNS', 'NNP', 'NNPS',
              'PDT', 'POS', 'PRP', 'PRP$',
              'RB', 'RBR', 'RBS', 'RP',
              'SYM', 'TO', 'UH',
              'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
              'WDT', 'WP', 'WP$', 'WRB']



###############################################################################
# Data sets...
###############################################################################

class GMB:
  """Provides access to the Groningen Meaning Bank dataset, as avaliable from https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus (the version with just tags, not the version with loads of extra calculated features). Interface to access sentence tokens is the same as the Tokenise class, but it adds the methods pos() and ner() to get the lists of labels."""
  
  def __init__(self, fn='ner_dataset.csv'):
    self._token = []
    self._pos = []
    self._ner = []
    
    with open(fn, newline='', encoding='ISO-8859-1') as fin:
      first = True
      for row in csv.reader(fin):
        if first:
          first = False
          continue
        
        if len(row)!=4:
          continue
        
        if row[0]!='':
          self._token.append([])
          self._pos.append([])
          self._ner.append([])
        
        self._token[-1].append(row[1])
        self._pos[-1].append(row[2])
        self._ner[-1].append(row[3])
        
          
  
  def __len__(self):
    return len(self._token)
  
  
  def __getitem__(self, index):
    return self._token[index]
  
  
  def pos(self, index):
    return self._pos[index]
  
  
  def ner(self, index):
    return self._ner[index]



###############################################################################
# Pretty printing...
###############################################################################

def aligned_print(*lines, line_width = 78):
  """For printing a sentence plus tags; you provide as many lists as you want and it aligns them horizontally, so tags appear under the word as aligned in the sentence. Also does colour using ansi escape codes (which work in Jupyter), because why not?"""
  codes = ['\x1b[0m', '\x1b[31m', '\x1b[34m', '\x1b[35m', '\x1b[36m']
  width = []
  blocks = [[]]
  total = 0
  
  for i in range(len(lines[0])):
    width.append(max(len(line[i]) for line in lines))
    
    if total + width[i] > line_width:
      blocks.append([])
      total = 0
    
    blocks[-1].append(i)
    total += width[i] + 1 # +1 for the space
  
  for block in blocks:
    for c, line in enumerate(lines):
      parts = []
      for i in block:
        parts.append(line[i])
        parts.append(' ' * (width[i] + 1 - len(line[i])))
      print(codes[c] + ''.join(parts))
    print(codes[0])
