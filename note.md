+ #### 数据驱动方法  
> + 图像处理任务：计算机视觉的核心任务  
> 当你做图像分类时，分类系统接受一些输入图像，并且系统清楚了一些已经确定的分类或者标签的集合，计算机的工作就是看图片，并且给它分配其中一些固定的分类标签  
> + The Problem: Semantic Gap(语义鸿沟)  
> 对于猫咪的概念或者它的标签，是我们赋给图像的一个语义标签，一只猫咪的语义概念和这些计算机实际看到的像素之间有着巨大的差距  
> + Challenges:   
>> + Viewpoint variation(视角):由于微小的方式改变图片，将导致像素网格整个发生变化，我们的算法需要对这些变化鲁棒  
>> + illumination(照明问题):  在场景中会有不同的照明条件，但在不同的光照下仍要识别出是同一只猫， 算法需要对这些变化鲁棒  
>> + Deformation(变形): 一只猫可能有千奇百怪的姿势和位置，对于不同的形变情形，我们的算法也应该是鲁棒的  
>> + Occlusion(遮挡)：  
>> + Background clutter(背景混乱):  
>> + Intraclass variation(类内差异): 一些猫因为颜色形态大小等，引起的视觉差异  
> **Data-Driven Approach(数据驱动方法):**
>> 1.利用搜索引擎抓取大量的相关图片数据集  
>> 2.训练机器来分类这些图片  ,总结生成一个模型，总结识别出这些不同类的对象的核心知识要素  
>> 3.利用这些模型识别新的图片  
>> 两个函数，一个是训练函数(接受图片和标签)，然后输入模型； 另一种是预测函数，接受模型，对图片种类进行预测  
> First classifier(分类器):**Nearest Neighbor(最近邻)**
> **L1距离(曼哈顿距离)：** 对图像的单个像素进行比较，用测试的图像减去训练图像的像素差的绝对值，所有差值相加求和(如果转动坐标轴，会改变点之间的L1距离)  
+ #### K-Nearest Neignbors(K-邻近算法)：  
>&emsp; 它不只找最最近的点，我们会做一些特殊操作，根据距离量度，找到最近的 K 个点，然后在这些相邻点中进行投票，预测出结果。(通常给K赋较大的值，这样会使决策边界更加平滑)  
> &emsp; **L2距离(欧氏距离)：** 取平方和的平方根，并且把这个作为距离(改变坐标轴对L2距离毫无影响)  
> &emsp; K 和距离度量的选择，称之为**超参数(hyperparameters)** , 无法从书籍中学习到，需要提前为算法做出选择  
+ ## 线性分类(Linear Classification)  
> **f(x, W) = Wx + b**  
>  &emsp;线性分类输入参数分类的一种，所有的训练数据中的经验知识都体现在参数矩阵 W 中，而 W 通过训练过程得到，我们拿到一张图，拉伸成一个长的向量，这里的图片假设叫做 x ，(例：拉伸成一个三维长向量(32, 32, 3) 其中高度和宽度是32像素，3则代表颜色通道 红、绿、 蓝)，还存在一个参数矩阵 W ，把这个代表图片像素的列向量当作输入，然后转化成10个数字评分  
> 线性分类可以解释为每个种类的学习模板  
+ ## 损失函数  
> 可以用一个函数把 W 当做输入，然后看一下得分，定量的估计 W 的好坏 这个函数被称为**损失函数**, 记作 L_i  
> + SVM loss:  
>>&emsp;对一个训练样例，若真实种类大于某一种类分数超过一个安全值，则这两种类的损失(loss)为 0 ；若不大于安全值，则求出差值并加上安全值，则为该两种类的损失值，将该样例与训练种类的损失值相加, 最终对于整个训练数据集的损失函数，是这些不同的样例损失函数的平均值    
>> + 标准损失函数拥有两个项，数据丢失项 和 正则项，这里有一些超参数 λ 用来平衡这两个项  
>> + 正则项是关于 W 的函数，为约束得到模型的唯一解，同时防止模型过拟合  
>> + L1度量复杂度的方式，有可能是非零元素的个数； 而L2更多的烤炉的是 W 整体布局，防止过拟合
> + doftmax loss(多项逻辑斯蒂回归)  
>> 用同 SVM 损失函数中算出的分数， 首先**指数化处理**(都变成正数),然后进行**归一化**(以保证他们的和为1)， L_i = -log(0.13) (0.13为真实情况对应的值)  
+ ## Optimization(优化)  
> + Strategy #1:随机采样，让后将他们输入损失函数
> + Strategy #2:Follow the slope  
> **梯度**  
> 找到 L 在 W 方向上的梯度  
> 沿着最陡的下降方向，即梯度的负方向，来一步步迭代，这样就能沿着损失函数从上往下走到最低点，走到了损失函数的等高轮廓的最低点  
> + 使用有限差分估计来计算数值梯度  
> + 使用解析梯度计数  
> + #### Mini-batch SGD(最小批量随机梯度下降)
>> &emsp;它的步骤是对数据进行连续的批量抽样，我们通过使用计算图或神经网络将数据进行正向传播，最终我们得到损失值，通过整个网络的反向传播来计算梯度，然后使用这个梯度来更新，网络中的参数或者权重。

+ ## 介绍神经网络-反向传播  
> + 计算图(框架): 我们用这类图来表示任意函数，其中图的节点，表示我们要执行的每一步计算
> + 反向传播是链式法则的递归调用，我们从计算图的最后面开始，从后往前，计算所有的梯度  
> + add gate: 加法节点连接了两个分支，获取上游梯度，并且知识分发和传递完全相同的梯度给相连的两个分支  
> + max gate: max 门将获取梯度，并且将其路由到它其中一个分支，另一分支梯度为 0  
> + mul gate: 获取上游梯度，然后根据另一个分支的值对其缩放  
> + 当有一个节点连接到多个节点时，梯度会在这个节点累加，得到其总上游梯度值    
+ ## 介绍神经网络-神经网络  
> + 神经网络就是：  
 &emsp;由简单函数构成的一组函数，在顶层堆叠在一起，我们用一种层次化的方式将它们堆叠起来，为了形成一个更复杂的非线性函数,多阶段分层计算   
 > + 激活函数，为了增加神经网络模型的非线性
+ ## 卷积神经网络  
> &emsp;**这是一种特殊的网络，它使用卷积层在贯穿整个网络的层次结构中，保持(输入)的空间结构，卷积层输出的每个激活图，是通过使用一个权重卷积核，在输入
(矩阵)的空间位置上滑动而生成的**
> + #### 卷积和池化  
>> 卷积层，它和全连接层(把32\*32\*3的 图，所有像素展开，得到一个3072维的向量)的主要差别，可以保全空间结构 ， 这里的权重时一些小的卷积核。  
>> + 我们的 ConvNet 基本上是由多个卷积层组成的一个序列，它们依次堆叠，就像我们之前在神经网络中，那样堆叠简单的线性层一样,之后我们将用激活函数对其进行逐一处理  
>> + 当你有了这些堆叠在一起的层时，你要知道它们是一些从简单到复杂的特征序列
>> + 卷积神经网络整体上来看  
>> &emsp; 是一个输入图片，让它通过很多层，第一个是**卷积层**( CONV ), 然后通常是**非线性层**( ReLU 就是一种非常常用的手段)，接下来会用到池化层( POOL )，这些措施已经大大降低了激活映射的采样尺寸，经过这些处理之后最终得到卷积层输出，然后我们就可以用我们之前见过的全连接层，连接所有的卷积输出，并用其获得一个最终的分值函数  
 >> + Output size:  
 >> **(N - F)/stride + 1**  
>> N 为输入的维度，F 为卷积核大小，滑动时的步幅为 stride  
>> **例：**  
>> N = 7, F = 3, stride = 1    
>> &emsp; 输出 5 \* 5  (最终输出 5 \* 5 \* 你使用的卷积核的数目)  
>> if 1 padded pixels( 0 补填像素)  
>> &emsp;实际七个卷积核都可以拟合，所以结果是一个 7 \* 7 的输出,这时的 N ≠ 7， N = 9  
>> + 做零填补的方式是，保持和我们之前的输入大小相同  
>> &emsp; 我们开始用的是7 \* 7，如果只是让卷积核从左上方角落处开始，将所有东西填入，那么之后我们会得到一个更小的输出，当我们会想保持全尺寸输出   
>> + **池化**：池化层所要做的就是要让所生成的表示更小且更容易控制，为了最后有更少的参数(降采样)  
&emsp;我们不会做在深度方向上的池化处理，而是只做平面上的，所以输入的深度和输出的深度是一样的  
> 计算公式同上 卷积(**但一般不再池化层做填零**)  
&emsp;最常见的方法是 **最大池化法**  
> 池化层也有一个卷积核的大小，而且卷积核的大小和我们所要池化处理的区域大小是相同的，不同的是这里不做数量积的计算，而是取该区域的最大值  
> **对于池化层，通常设定步长，使它们不会互相重叠**
> + #### 视觉之外的卷积神经网络  
>> + 有一个 5 \* 5 的卷积核，我们也可以称它为这个神经元的一个 5 \* 5的感受野(receptive field)，  
+ ## 训练神经网络  
> + #### 激活函数  
>> &emsp;我们输入数据，在全连接层或者卷积层，我们将输入乘上权重值，然后将结果输入一个激活函数或者非线性(单元)  
>> + **sigmoid 函数**   
>>>![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/1.png)  
>>> &emsp; 每个元素被输入到 sigmoid 非线性函数中，每个元素被压缩到[0,1]范围内，输入+∞，输出将无限趋近于1；输入-∞，输出将无限接近于0，在横坐标接近于0的区域中，我们可以将这部分看作是线性区域  
>>> &emsp; sigmoid 函数在某种意义上，可以被看作是一种神经元的饱和放电率  
>>> + 3 problems:  
>>> &emsp; 1.当输入 x 等于一个很大的负值或很大的正值时，它们位于 sigmoid 函数的平滑区域，这些区域会使梯度消失，从而无法得到梯度流的反馈  
>>> &emsp; 2.sigmoid outputs are not zero-centered  
>>> &emsp; 3.exp() is a bit compute expensive  
>> + **tanh 激活函数**  
>>>![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/2.png)  
>>> &emsp; 和 sigmoid 函数很相似，不同在于， tanh 被挤压到[-1,1]的范围内  
>>> &emsp; tanh 函数是以 0 为中心的(所以不会有 sigmoid 函数的第二个问题)  
>>> &emsp; 当它饱和时依然会出现梯度消失问题   
>> + **ReLU 激活函数**  
>>>![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/3.png)  
>>> f(x) = max(0,x)   
>>> &emsp; 若输入为负数，输出为 0；  若输入为正数，输出为其本身  
>>> + advantages:   
>>> &emsp;  x为正时不会存在饱和;   
>>> &emsp; 计算成本不高；   
>>> &emsp; 收敛速度比上两个快大约 6 倍；   
>>> &emsp; 比 sigmoid 更具生物学的合理性  
>>> + problems:  
>>> 不是以 0 为中心的  
>>> 虽然正半轴不产生饱和，但是负半轴饱和  
>> + **Leaky ReLU 激活函数**  
>>>![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/4.png)  
>>> f(x) = max(0.01x, x)  
>>> + advantages:    
>>> &emsp; x为正时不会存在饱和;   
>>> &emsp; 计算成本不高；   
>>> &emsp; 收敛速度 sigmoid/tanh 上两个快大约 6 倍；  
>>> &emsp; will not "die"  
>>> + **&emsp;Parametric Rectifier(PReLU) 参数整流器:**  f(x) = max(αx, x)  
>> + **指数线性单元 (ELU)**  
>>>![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/5.png)  
>>> + advantages:  
>>> &emsp; 具有 ReLU 的所有优点  
>>> &emsp; 输出均值接近为 0  
>>> &emsp; 和 ReLU 相比，ELU 没有在负区间倾斜，一些有争议的观点： 这样使得模型对噪音具有更强的鲁棒性  
>> + In practice:
>>> + Use <font color = #7FFFD4>ReLU</font>. Be careful with your learning rates  
>>> + Try out <font color = #DEB887>Leaky ReLU/ Maxout/ ELU</font>  
>>> + Try out <font color = #ff0000>tanh</font> but don't expect much    
>>> + <font color = #ff0000>Don't use sigmoid</font>  
> + #### 初始化权重  
>> + First idea: **Small random numbers**(适用于小型网络)
>> + 在开始训练时，初始化的权重值(即 W 参数)  
>> + 如果权重太小，在学习深度网络时，激活值会消失；  
>> + 如果权重初始值过大 ，那么这些初始值不断地乘以你的权值矩阵，将会爆炸增长  
>> &emsp;**Reasonable initialization:**  Xavier初始化法 initialization  
>> ![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/6.png)  
>> &emsp;ReLU 由于有一般的神经元被置 0 ，(和之前未用 ReLU 激活函数相比)等效的输入，实际只有一半的输入，所以只需要除以 2 这个因子， done.
>> ![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/7.png)
> + #### 数据预处理  
>> &emsp; 在卷积神经网络中，中心化和归一化是非常常用的手段，它会使数据分布均值为零，方差为一  
>> &emsp; 使用归一化，我们的损失函数对参数值中地小扰动就不那么敏感了； 如果神经网络中，某一层地输入均值不为 0 ，或者方差不为 1，该层网络权值矩阵地微小摄动，都会造成该层输出的巨大摄动    
>> + **batch normalization(批量归一化)**  
>> &emsp; 即在神经网络中加入额外一层以使得中间的激活值均值为0方差为1  
>> &emsp;  在batch normalization中，正向传递时，我们用小批量地统计数据计算平均值和标准差，并用这个估计值并且对数据进行归一化，同时还有缩放函数和平移函数来增加一层地可表达性   
> + #### 监督学习  
>> ![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/8.png)  
> + #### hyperparameter search(超参数搜索)  
>> + 网格搜索，随机搜索，当你的模型性能对某一个超参数比对其他超参数更敏感的时候，随机搜索可以对超参数空间覆盖的更好  
>> + 粗细粒交叉搜索，当你做超参数优化的时候，一开始可能会处理很大的搜索范围，几次迭代后，就可以缩小范围，圈定合适的超参数所在的区域，然后在对这个小范围，重复这个过程，可以多次迭代上述步骤，以获得超参数的正确区域  
> + #### Fancier Optimization(更好的优化)   
>>  + **随机梯度下降(SGD)**  
>>> 首先评估一下一些小批数据中损失的梯度，更进一步，向梯度为负的方向更新参数向量，重复这个过程，它在红色区域收敛，得到很小的误差值  
>>> + problem one： ![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/9.png)  
>>> + problem two:  
>>> &emsp; **局部极小值**或**鞍点**，会卡在那里  
>>> &emsp; 在一维问题上，局部极小值看起来是个大问题，鞍点看起来不需要担心； 在高维空间相反  
>>> + problem three:  
>>> &emsp; 随机性，我们通常使用小批量的计算对损失和梯度进行评估
>>
>> + **SGD + Momentum(加动量)**  
>>> &emsp; 保持一个不随时间变化的速度，并且我们将梯度估计添加到这个速度上，然后在这个速度的方向上步进，而不是在梯度的方向上步进  
>>> &emsp; 类似于惯性，使得运动到局部极小值点或者鞍点时，虽然剃度为0，但是仍存在一个速度得以继续前进  
>>> &emsp; 有时会看到动量的一个轻微变化，叫做 Nesterov 加速梯度(动量)  
>> + **Nesterov Momentum**  
>>> ![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/10.png)   
>>>&emsp; 现在更新速度，根据之前的速度来前进一步，然后计算此处的梯度，现在，当我们前进下一步时，我们实际是在速度的方向上步进，
换元后， Nesterov 可以同时计算损失函数和梯度；   
>>> &emsp; Nesterov 动量包含了当前速度向量和先前速度向量的误差修正  
>> + **AdaGrad 算法**
>>> &emsp; 你在优化的过程中，需要保持一个在训练过程中的每一步的梯度的平方和的持续估计，与速度项不同的是，现在有了一个梯度平方项，在训练时，我们会一直累加当前梯度的平方到这个梯度平方项，当我们更新参数向量时，我们会除以这个梯度平方项  
>> + **RMSProp 算法**  
>>>![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/11.png)   
>>> &emsp; 在 RMSProp 中，我们仍然计算梯度的平方，但是并不是仅仅简单的在训练中累加梯度平方，而是会让平方梯度按照一定比率下降，类似于动量优化法，我们是给梯度的平方加上动量，而不是给梯度本身  
>>> &emsp; 在计算完梯度后，取出当前的梯度平方，将其乘以一个衰减率，然后用1减去衰减率乘以梯度的平方加上之前的结果  
>> + ** 接近Adam 的算法**  
>>>  ![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/12.png)   
>>> &emsp;使用 Adam 算法更新第一动量和第二动量的估计值，在红框里，我们让第一动量的估计值等于我们梯度的加权和，我们有一个第二动量的动态估计值，一个梯度平方的动态近似值  
>>> &emsp; 使用第一动量，有点类似于速度，并且除以第二动量，或者说第二动量的平方根，就是这个梯度的平方项  
>>> &emsp; problem:在最初的第一步会得到一个非常大的步长，是因为我们人为的把第二动量设为0造成的  
>> + **Adam(full form)**  
>>> ![](https://github.com/W-Avan/Machine_Learning/raw/master/pic/13.png)   
>>> 在我们更新了第一和第二动量之后，构造了第一和第二动量的无偏估计，通过使用当前时间t，现在实际上在使用无偏估计来做每一步更新，而不是初始的第一和第二动量的估计值
> + #### Regularization(正则化)  
>>  
> + #### Transfer Learning(迁移学习)  
>>  