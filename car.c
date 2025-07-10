#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <errno.h>
#include <signal.h>

// 电机控制引脚定义 (使用实际的GPIO编号)
#define MOTOR_A_IN1_GPIO 115   // 物理引脚33 - 左电机控制引脚1
#define MOTOR_A_IN2_GPIO 123   // 物理引脚35 - 左电机控制引脚2
#define MOTOR_B_IN1_GPIO 114   // 物理引脚37 - 右电机控制引脚1
#define MOTOR_B_IN2_GPIO 104   // 物理引脚12 - 右电机控制引脚2

// 串口设备 - RK3588的UART9
#define UART_DEV "/dev/ttyS9"

// PWM相关定义 - 左右电机共用一个PWM通道
#define PWM_CHIP        "0"       // PWM芯片编号
#define PWM_SHARED      "0"       // 共用的PWM通道（路径：/sys/class/pwm/pwmchip0/pwm0）
#define PWM_PERIOD_NS   20000000  // 20ms周期（50Hz）

// 全局文件描述符
int uart_fd = -1;            // 串口文件描述符
int gpio_fds[4] = {-1, -1, -1, -1}; // GPIO值文件描述符数组
int pwm_shared_fd = -1;      // 共用PWM通道文件描述符

// 全局状态变量
int current_speed = 80;      // 当前速度百分比 (0-100)

// 函数声明
int init_uart();
bool init_gpio();
void set_gpio_value(int index, int value);
bool control_car(char command);
void stop_motors();
void cleanup(int signum);
bool init_shared_pwm();
void set_pwm_duty(int fd, int percent);
void cleanup_shared_pwm();
void log_message(const char* message);

// 信号处理函数 - 捕获终止信号并执行资源清理
void signal_handler(int signum) {
    log_message("接收到终止信号，准备清理资源...");
    cleanup(signum);
    exit(signum);
}

// 日志函数 - 统一处理日志输出
void log_message(const char* message) {
    printf("[INFO] %s\n", message);
}

// 初始化串口通信
int init_uart() {
    // 打开串口设备
    int fd = open(UART_DEV, O_RDWR | O_NOCTTY);
    if (fd == -1) {
        perror("[ERROR] 无法打开串口设备");
        return -1;
    }
    
    // 配置串口参数
    struct termios options;
    if (tcgetattr(fd, &options) < 0) {
        perror("[ERROR] 获取串口属性失败");
        close(fd);
        return -1;
    }
    
    // 设置波特率为9600
    cfsetispeed(&options, B9600);
    cfsetospeed(&options, B9600);
    
    // 配置数据位、停止位和校验位
    options.c_cflag &= ~PARENB;    // 无校验
    options.c_cflag &= ~CSTOPB;    // 1位停止位
    options.c_cflag &= ~CSIZE;     // 清除数据位掩码
    options.c_cflag |= CS8;        // 8位数据位
    
    // 禁用硬件流控制
    options.c_cflag &= ~CRTSCTS;
    
    // 启用接收和本地模式
    options.c_cflag |= (CLOCAL | CREAD);
    
    // 禁用软件流控制
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    
    // 设置原始输入/输出模式
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    
    // 设置超时参数
    options.c_cc[VMIN]  = 0;      // 最小读取字符数
    options.c_cc[VTIME] = 1;      // 超时时间 (0.1秒)
    
    // 刷新缓冲区并应用配置
    tcflush(fd, TCIFLUSH);
    if (tcsetattr(fd, TCSANOW, &options) < 0) {
        perror("[ERROR] 设置串口属性失败");
        close(fd);
        return -1;
    }
    
    log_message("串口初始化成功");
    return fd;
}

// 导出GPIO并设置方向
bool export_and_set_gpio(int gpio_num) {
    char path[50];
    char buffer[10];
    int fd;
    
    // 导出GPIO
    fd = open("/sys/class/gpio/export", O_WRONLY);
    if (fd < 0) {
        perror("[ERROR] 无法打开GPIO导出文件");
        return false;
    }
    
    sprintf(buffer, "%d", gpio_num);
    if (write(fd, buffer, strlen(buffer)) < 0) {
        if (errno != EBUSY) {  // 忽略已导出的错误
            perror("[ERROR] 导出GPIO失败");
            close(fd);
            return false;
        }
    }
    close(fd);
    
    // 等待GPIO目录创建
    usleep(100000); // 100ms
    
    // 设置GPIO为输出模式
    snprintf(path, sizeof(path), "/sys/class/gpio/gpio%d/direction", gpio_num);
    fd = open(path, O_WRONLY);
    if (fd < 0) {
        perror("[ERROR] 无法打开GPIO方向文件");
        return false;
    }
    
    if (write(fd, "out", 3) < 0) {
        perror("[ERROR] 设置GPIO方向失败");
        close(fd);
        return false;
    }
    close(fd);
    
    return true;
}

// 初始化GPIO引脚
bool init_gpio() {
    int gpio_nums[4] = {
        MOTOR_A_IN1_GPIO,
        MOTOR_A_IN2_GPIO,
        MOTOR_B_IN1_GPIO,
        MOTOR_B_IN2_GPIO
    };
    
    // 导出并设置所有GPIO引脚
    for (int i = 0; i < 4; i++) {
        if (!export_and_set_gpio(gpio_nums[i])) {
            fprintf(stderr, "[ERROR] 初始化GPIO引脚 %d 失败\n", gpio_nums[i]);
            return false;
        }
    }
    
    // 打开GPIO值文件并保存描述符
    for (int i = 0; i < 4; i++) {
        char path[50];
        snprintf(path, sizeof(path), "/sys/class/gpio/gpio%d/value", gpio_nums[i]);
        gpio_fds[i] = open(path, O_WRONLY);
        if (gpio_fds[i] < 0) {
            perror("[ERROR] 无法打开GPIO值文件");
            return false;
        }
    }
    
    // 初始状态：停止所有电机
    stop_motors();
    log_message("GPIO初始化成功");
    return true;
}

// 设置GPIO输出值
void set_gpio_value(int index, int value) {
    if (index < 0 || index >= 4 || gpio_fds[index] < 0) {
        fprintf(stderr, "[ERROR] 无效的GPIO索引或文件描述符\n");
        return;
    }
    
    char val = value ? '1' : '0';
    if (write(gpio_fds[index], &val, 1) < 0) {
        perror("[ERROR] 设置GPIO值失败");
    }
}

// 控制小车运动
bool control_car(char command) {
    bool valid_command = true;
    
    switch (command) {
        case 'f': // 前进
            set_gpio_value(0, 1); // 左电机正转
            set_gpio_value(1, 0);
            set_gpio_value(2, 1); // 右电机正转
            set_gpio_value(3, 0);
            log_message("前进");
            break;
            
        case 'b': // 后退
            set_gpio_value(0, 0); // 左电机反转
            set_gpio_value(1, 1);
            set_gpio_value(2, 0); // 右电机反转
            set_gpio_value(3, 1);
            log_message("后退");
            break;
            
        case 'l': // 左转
            set_gpio_value(0, 0); // 左电机反转
            set_gpio_value(1, 1);
            set_gpio_value(2, 1); // 右电机正转
            set_gpio_value(3, 0);
            log_message("左转");
            break;
            
        case 'r': // 右转
            set_gpio_value(0, 1); // 左电机正转
            set_gpio_value(1, 0);
            set_gpio_value(2, 0); // 右电机反转
            set_gpio_value(3, 1);
            log_message("右转");
            break;
            
        case 's': // 停止
            stop_motors();
            log_message("停止");
            break;
            
        default:
            log_message("未知命令");
            valid_command = false;
            break;
    }
    
    return valid_command;
}

// 停止所有电机
void stop_motors() {
    // 关闭所有电机控制引脚
    for (int i = 0; i < 4; i++) {
        set_gpio_value(i, 0);
    }
    
    // 停止时降低PWM占空比以节省能耗
    set_pwm_duty(pwm_shared_fd, 0);
}

// 初始化共用PWM通道
bool init_shared_pwm() {
    char path[64], buf[16];
    int tmpfd;
    
    // 导出PWM通道（路径：/sys/class/pwm/pwmchip0/pwm2）
    snprintf(path, sizeof(path), "/sys/class/pwm/pwmchip%s/export", PWM_CHIP);
    tmpfd = open(path, O_WRONLY);
    if (tmpfd >= 0) {
        write(tmpfd, PWM_SHARED, strlen(PWM_SHARED));
        close(tmpfd);
        usleep(100000); // 等待节点创建
    } else {
        perror("[ERROR] 导出共用PWM通道失败");
        return false;
    }
    
    // 设置PWM周期
    snprintf(path, sizeof(path), "/sys/class/pwm/pwmchip%s/pwm%s/period", 
             PWM_CHIP, PWM_SHARED);
    tmpfd = open(path, O_WRONLY);
    if (tmpfd >= 0) {
        snprintf(buf, sizeof(buf), "%d", PWM_PERIOD_NS);
        write(tmpfd, buf, strlen(buf));
        close(tmpfd);
    } else {
        perror("[ERROR] 设置PWM周期失败");
        return false;
    }
    
    // 设置初始占空比为0
    snprintf(path, sizeof(path), "/sys/class/pwm/pwmchip%s/pwm%s/duty_cycle", 
             PWM_CHIP, PWM_SHARED);
    tmpfd = open(path, O_WRONLY);
    if (tmpfd >= 0) {
        snprintf(buf, sizeof(buf), "%d", 0);
        write(tmpfd, buf, strlen(buf));
        close(tmpfd);
    } else {
        perror("[ERROR] 设置PWM初始占空比失败");
        return false;
    }
    
    // 使能PWM输出
    snprintf(path, sizeof(path), "/sys/class/pwm/pwmchip%s/pwm%s/enable", 
             PWM_CHIP, PWM_SHARED);
    tmpfd = open(path, O_WRONLY);
    if (tmpfd >= 0) {
        write(tmpfd, "1", 1);
        close(tmpfd);
    } else {
        perror("[ERROR] 启用PWM输出失败");
        return false;
    }
    
    // 打开duty_cycle文件用于后续快速调节
    snprintf(path, sizeof(path), "/sys/class/pwm/pwmchip%s/pwm%s/duty_cycle", 
             PWM_CHIP, PWM_SHARED);
    pwm_shared_fd = open(path, O_WRONLY);
    if (pwm_shared_fd < 0) {
        perror("[ERROR] 打开PWM占空比文件失败");
        return false;
    }
    
    log_message("共用PWM通道初始化成功");
    return true;
}

// 设置PWM占空比 (0-100%)
void set_pwm_duty(int fd, int percent) {
    if (fd < 0) return;
    
    // 限制百分比范围
    int clamped_percent = percent;
    if (clamped_percent < 0) clamped_percent = 0;
    if (clamped_percent > 100) clamped_percent = 100;
    
    // 计算实际占空比值 (纳秒)
    int duty = PWM_PERIOD_NS * clamped_percent / 100;
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", duty);
    
    // 重置文件指针并写入新占空比
    if (lseek(fd, 0, SEEK_SET) == -1) {
        perror("[ERROR] 重置PWM文件指针失败");
        return;
    }
    
    if (write(fd, buf, strlen(buf)) == -1) {
        perror("[ERROR] 设置PWM占空比失败");
    }
}

// 清理共用PWM资源
void cleanup_shared_pwm() {
    if (pwm_shared_fd >= 0) {
        // 先设置占空比为0再关闭
        set_pwm_duty(pwm_shared_fd, 0);
        close(pwm_shared_fd);
        pwm_shared_fd = -1;
    }
    
    // 取消导出PWM通道
    char path[64];
    snprintf(path, sizeof(path), "/sys/class/pwm/pwmchip%s/unexport", PWM_CHIP);
    int tmpfd = open(path, O_WRONLY);
    if (tmpfd >= 0) {
        write(tmpfd, PWM_SHARED, strlen(PWM_SHARED));
        close(tmpfd);
    }
}

// 清理所有资源
void cleanup(int signum) {
    log_message("开始资源清理...");
    
    // 停止所有电机
    stop_motors();
    
    // 清理PWM资源
    cleanup_shared_pwm();
    
    // 关闭GPIO文件描述符
    for (int i = 0; i < 4; i++) {
        if (gpio_fds[i] >= 0) {
            close(gpio_fds[i]);
            gpio_fds[i] = -1;
        }
    }
    
    // 关闭串口
    if (uart_fd >= 0) {
        close(uart_fd);
        uart_fd = -1;
    }
    
    log_message("资源清理完成");
}

int main() {
    log_message("蓝牙小车控制器启动中...");
    
    // 设置信号处理
    signal(SIGINT, signal_handler);  // 处理Ctrl+C
    signal(SIGTERM, signal_handler); // 处理系统终止信号
    
    // 初始化串口
    uart_fd = init_uart();
    if (uart_fd < 0) {
        fprintf(stderr, "[FATAL] 串口初始化失败，程序退出\n");
        return 1;
    }
    
    // 初始化GPIO
    if (!init_gpio()) {
        fprintf(stderr, "[FATAL] GPIO初始化失败，程序退出\n");
        cleanup(0);
        return 1;
    }
    
    // 初始化共用PWM通道
    if (!init_shared_pwm()) {
        fprintf(stderr, "[FATAL] 共用PWM通道初始化失败，程序退出\n");
        cleanup(0);
        return 1;
    }
    
    // 设置初始速度
    set_pwm_duty(pwm_shared_fd, current_speed);
    log_message("蓝牙小车控制器已就绪，等待命令...");
    
    // 主循环 - 读取串口命令并控制小车
    char buffer[128];
    while (1) {
        ssize_t count = read(uart_fd, buffer, sizeof(buffer) - 1);
        if (count > 0) {
            buffer[count] = '\0'; // 确保字符串终止
            printf("[CMD] 收到命令: %s\n", buffer);
            
            // 解析命令中的速度参数（如f80、b60等）
            int new_speed = -1;
            if (count >= 2 && sscanf(buffer+1, "%d", &new_speed) == 1) {
                // 验证速度范围
                if (new_speed >= 0 && new_speed <= 100) {
                    current_speed = new_speed;
                    set_pwm_duty(pwm_shared_fd, current_speed);
                    printf("[INFO] 速度已设置为 %d%%\n", current_speed);
                } else {
                    printf("[ERROR] 无效的速度值: %d (范围应为0-100)\