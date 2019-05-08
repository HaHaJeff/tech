# paxos

paxos作为一种分布式环境下的共识性算法，其目标是保证在多数派机器存活的情况下依旧能够维持数据的一致性，其与2pc等算法的区别在于：

- 2PC属于阻塞的且不容错的
- paxos属于非阻塞且容错的

## prepare阶段

- proposer向acceptor广播prepare请求，prepare请求需要携带上proposalId号
  - acceptor接收到prepare请求，如果proposalId大于acceptor接受过的最大proposalId，则同意：并作出如下两个承诺，并返回其接受过最高的proposalId的value（如果有的话），置minProposalId=proposalId
    - 不再接受小于等于proposalId的prepare请求
    -  不再接受小于proposalId的accept请求
- 如果proposer接受到多数派的成功响应，就可发送accept请求了
  - accept请求由proposalId和value构成，value由prepare阶段中最高proposalId的value决定

## learners

学习一个已经被choosen的proposals

- 每一个acceptor都向learner发送已经被choosen的proposal
  - 有效但是开销大
- 选择一个法定learner
  - 一个被选定的proposal只有proposer知道，所以当proposal被choosen时，proposer将该proposal发送给所有learners（acceptor）

## 面试中遇到的问题

- proposalId能否一样

  论文中提到：paxos正确的一个前提就是说：每个proposer需要不一样的proposalId

  **phxpaxos也是这样实现的**

# raft

- follower
- candidater
- leader

**raft采用强leader保证算法的简洁性以及正确性**

## leader选举

**raft使用心跳出发leader选举。当服务器启动时，初始化为follower。leader向followers周期性发送heartBeat。如果follower在选举超时时间内没有收到leader的heartBeat，就会等待一段随机的时间后发起一次leader选举**

随机等待的目的是方式选举冲突。

- 只有包含最新log的candidater才能竞选成功，raft如何保证？
  - RequestVote会携带prevLogIndex以及prevLogTerm，其他节点在收到消息时：
    - 如果发现自己的日志比请求中携带的更新，则拒绝投票。比较原则：如果本地的最后一条的log entry的term更大，则term大的更新。如果term一样大，则log index更大的更新
    - 否则，同意该candidater的选举

## 日志复制

leader把客户端请求作为日志条目，然后并行的向其他服务器发起AppendEntries RPC复制条目。当这条日志被复制到大多数服务器上时，leader将这条日志应用到它的状态机中，并向client返回结果。

raft日志同步保证以下两点：

- 如果不同日志中的两个条目有着相同的索引号和任期号，则它们所存储的命令相同
- 如果不同日志中的两个条目有着相同的索引号和任期号，则它们之前的所有条目都是一样的

第一特性源于leader在一个term内，在一个给定的log index最多创建一条日志条目，同时该条目在日志中的位置也从来不会改变

第二条特性源于AppendEntries的一个简单的一致性检查。当发送一个AppendEntries RPC时，Leader会把新日志条目紧接着之前的条目的log index和term都包含在里面。如果Follower没有在它的日志中找到log index和term都相同的日志，它就会拒绝新的日志条目。

- leader通过强制followers复制它的日志来处理日志的不一致，followers上的不一致的日志会被leader的日志覆盖。
- leader为了使followers的日志同自己的一致，leader需要找到followers同它的日志一致的地方，然后覆盖followers在该位置之后的条目。
- leader会从后往前尝试，每次AppendEntries失败后尝试前一个日志条目，直到成功找到每个follower的日志一致位点，然后向后覆盖

## 安全性

- 拥有最新的已提交的log entry的follower才能成为leader
- leader只能推进commit index来提交当前term，旧term日志的提交要等到当前term的日志间接提交