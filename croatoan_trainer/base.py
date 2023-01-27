class Base():
    def set_plotly_args(self, **kwargs):
        """
        Sets plotly args for charts

        Args:
            `**kwargs`: named arguments for plotly `update_layout()` method
            (name of arguments must match arguments from this method).
            Example: `font_size=16, template='plotly_dark'`
        """
        self.plotly_args = kwargs
