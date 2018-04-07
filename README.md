# chinese-poem-generator-with-char-rnn

[chinese-poem-generator](https://github.com/linpingta/chinese-poem-generator)项目的char rnn版本，代码大部分来自[char-rnn-tensorflow](https://github.com/linpingta/char-rnn-tensorflow)，针对中文宋词的生成，做了部分修改。

【背景】
在16年参加公司黑客马拉松项目时，基于规则做了一版诗词生成，也取得了不错的结果。但当时由于时间限制，没有尝试深度学习相关的方案，在这里算是一个补充。

【远离】
如果已经阅读过char-rnn-tensorflow的代码，本项目中大部分代码阅读没有障碍。我所做的工作，主要是在char-rnn-tensorflow基础上，解决它对于句子定长的限制（这个描述不完全准确，它实际是在生成一篇文章，因此要求设置seq_length，但在宋词生成的方面，句子和句子之间有不同的含义，因此基于每个句子去做seq2seq翻译，似乎更合理一些）。主要区别在网络结构定义和损失函数上，用dynamic_rnn替代legacy_seq2seq.rnn_decoder, 用softmax_cross_entropy_with_logits替代legacy_seq2seq.sequence_loss_by_example。

另外针对宋词生成押韵的要求，我在生成训练样本时是倒序生成的，这样的好处是可以把韵脚作为一句话的输入，对于不同词牌，采用不同押韵方式，可以并行生成一首词。

【运行】
训练部分：

    python main.py --num_epochs 50

生成数据保存在save中，可以通过tensorboard查看log中内容

生成部分：
1. 生成一句话

    python inference.py --generate_type one --word_len 5 --input_word 明
    
    result: words  巫山清露明

2. 生成一个词牌结果

    python inference.py --generate_type multi --title 采桑子

    result: 苕渡、忆山霾深，荣老花同，诗人客啸，高共醉举郎书巧，当著旧衰小庭暝，鹏思旧同，波臻俳峭，似彼花开有开拗

【其它】
目前效果大部分都不能beat[chinese-poem-generator](https://github.com/linpingta/chinese-poem-generator)项目的结果，可能有两方面的原因
1. 规则系统缺失，有些标点符号不该生成的地方没有处理
2. 可能模型没有收敛，训练数据还是比较少
作为一个toy项目，还是可以使用的 :)
