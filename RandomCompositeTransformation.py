import random
from textattack.transformations.composite_transformation import CompositeTransformation

class RandomCompositeTransformation(CompositeTransformation):
    """
    允许以一定概率执行多个数据增强方法的组合转换。
    """

    def __init__(self, transformations, probabilities):
        """
        :param transformations: 一个列表，包含多个 Transformation 对象（如 WordNet 替换、Embedding 替换）。
        :param probabilities: 一个列表，包含每个转换被选择的概率（总和不需要为 1）。
        """
        assert len(transformations) == len(probabilities), "transformations 和 probabilities 长度必须相同"
        super().__init__(transformations)
        self.probabilities = probabilities

    def _get_transformations(self, current_text, indices_to_modify=None):
        """按照概率随机选择要执行的转换"""
        selected_transformations = [
            t for t, p in zip(self.transformations, self.probabilities) if random.random() < p
        ]
        if not selected_transformations:
            selected_transformations = [random.choice(self.transformations)]  # 确保至少执行一个
        
        # 应用选中的转换
        transformed_texts = []
        for transformation in selected_transformations:
            transformed_texts.extend(transformation(current_text, indices_to_modify))
        return transformed_texts