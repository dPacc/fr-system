module.exports = {
  module: {
    rules: [
      {
        test: /\.js$/,
        use: ["source-map-loader"],
        enforce: "pre",
        exclude: /node_modules\/face-api.js/, // Exclude face-api.js from source map loading
      },
    ],
  },
};
