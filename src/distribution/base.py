from abc import ABC, abstractmethod


# base class for distribution
class distribution(object):
	def __init__(self, transformation, name, **kwargs):
		self.transformation = transformation
		self.name = name

	@abstractmethod
	def sample(self, Input):
		pass
		
	@abstractmethod
	def sample_and_log_prob(self, Input):
		pass

	@abstractmethod
	def log_prob(self, Input, output):
		pass

	@abstractmethod
	def mean(self, Input, output):
		pass

