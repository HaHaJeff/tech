- 写时复制技术允许父子进程读相同的物理页。只要两者中有一个试图写一个物理页，内核就把这个页的内容拷贝到一个新的物理页，并把这个物理页分配给正在写的进程。
- 轻量级进程允许父子进程共享每进程在内核的很多数据结构，如页表，打开文件表以及信号处理
- vfork系统调用创建的进程能够共享其父进程的内存地址空间。为了防止父进程重写子进程需要的数据，阻塞父进程的执行，一直到子进程退出或执行一个新的程序为止。

# fork

1. 获得一个pid，从pid_maps中
2. copy_process
   1. 分配task_struct描述符
   2. 分配thread_info描述符   thread_info **与进程描述符相关的小数据结构，叫做线程描述符，存放在内核栈中**
   3. 为上述结构赋值（都是指向current的相应字段）  
   4. copy_semundo
   5. copy_fs
   6. copy_file
   7. copy_mm   （如果有CLONE_VM标志的，就不会allocate struct 结构，而是直接将new mm指针指向current->mm，所以vfork在mutex中可以瞎玩，但是fork不应，因为mutex以mm地址为key建立mutex）
   8. 各种copy

子进程不继承

- 进程号，子进程号不同于任何一个活动的进程号
- 父进程号
- 子进程不继承父进程的记录锁
- pend signals也不会被继承
- 与CPU始终相关的会置0

# fork vfork clone

这三个系统调用最后殊途同归，最后都是经过do_fork，如果fork流程差不多，只不过flag有些许不同