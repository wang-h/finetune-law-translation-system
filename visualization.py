import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import threading
import time
from collections import deque
import seaborn as sns

import matplotlib.font_manager as fm

# 尝试加载当前目录下的 SimHei 字体
font_path = 'SimHei.ttf'
if os.path.exists(font_path):
    # 加载自定义字体
    my_font = fm.FontProperties(fname=font_path)
    # 将 SimHei 添加到字体管理器
    fm.fontManager.addfont(font_path)
    # 设置为默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
else:
    # 备用设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'WenQuanYi Micro Hei']

plt.rcParams['axes.unicode_minus'] = False

class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, output_dir="./training_logs", enable_realtime=True, 
                 enable_tensorboard=False, max_points=1000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.enable_realtime = enable_realtime
        self.enable_tensorboard = enable_tensorboard
        self.max_points = max_points
        
        # 训练数据存储
        self.train_losses = deque(maxlen=max_points)
        self.val_losses = deque(maxlen=max_points)
        self.learning_rates = deque(maxlen=max_points)
        self.steps = deque(maxlen=max_points)
        self.epochs = deque(maxlen=max_points)
        
        # 实时绘图设置
        if self.enable_realtime:
            self.setup_realtime_plot()
        
        # TensorBoard设置
        if self.enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
                print(f"✅ TensorBoard日志目录: {self.output_dir / 'tensorboard'}")
                print("   启动命令: tensorboard --logdir=./training_logs/tensorboard")
            except ImportError:
                print("⚠️  TensorBoard不可用，请安装: pip install tensorboard")
                self.enable_tensorboard = False
        
        # 训练统计
        self.start_time = datetime.now()
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def setup_realtime_plot(self):
        """设置实时绘图"""
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        
        # 创建图形和子图
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training Monitor', fontsize=16, fontweight='bold')
        
        # 子图标题和标签
        self.ax1.set_title('Training & Validation Loss', fontweight='bold')
        self.ax1.set_xlabel('Steps')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Learning Rate', fontweight='bold')
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Learning Rate')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('Loss per Epoch', fontweight='bold')
        self.ax3.set_xlabel('Epochs')
        self.ax3.set_ylabel('Avg Loss')
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_title('Training Stats', fontweight='bold')
        self.ax4.axis('off')
        
        # 初始化线条
        self.train_line, = self.ax1.plot([], [], 'b-', label='Train Loss', linewidth=2)
        self.val_line, = self.ax1.plot([], [], 'r-', label='Val Loss', linewidth=2)
        self.lr_line, = self.ax2.plot([], [], 'g-', label='Learning Rate', linewidth=2)
        
        self.ax1.legend()
        self.ax2.legend()
        
        # 调整布局
        plt.tight_layout()
        plt.ion()  # 开启交互模式
        plt.show()
    
    def log_step(self, step, epoch, train_loss, val_loss=None, learning_rate=None):
        """记录训练步骤数据"""
        self.steps.append(step)
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        # TensorBoard记录
        if self.enable_tensorboard:
            self.tb_writer.add_scalar('Loss/Train', train_loss, step)
            if val_loss is not None:
                self.tb_writer.add_scalar('Loss/Validation', val_loss, step)
            if learning_rate is not None:
                self.tb_writer.add_scalar('Learning_Rate', learning_rate, step)
    
    def update_plot(self):
        """更新实时图表"""
        if not self.enable_realtime or len(self.steps) == 0:
            return
        
        try:
            # 更新损失曲线
            steps_list = list(self.steps)
            train_losses_list = list(self.train_losses)
            
            self.train_line.set_data(steps_list, train_losses_list)
            
            if len(self.val_losses) > 0:
                val_losses_list = list(self.val_losses)
                self.val_line.set_data(steps_list[-len(val_losses_list):], val_losses_list)
            
            # 更新学习率曲线
            if len(self.learning_rates) > 0:
                lr_list = list(self.learning_rates)
                self.lr_line.set_data(steps_list[-len(lr_list):], lr_list)
            
            # 自动调整坐标轴
            if len(steps_list) > 1:
                self.ax1.relim()
                self.ax1.autoscale_view()
                self.ax2.relim()
                self.ax2.autoscale_view()
            
            # 更新epoch损失对比
            self.update_epoch_comparison()
            
            # 更新统计信息
            self.update_stats_display()
            
            # 刷新显示
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"绘图更新错误: {e}")
    
    def update_epoch_comparison(self):
        """更新每轮损失对比"""
        if len(self.epochs) == 0:
            return
        
        # 计算每个epoch的平均损失
        epoch_train_losses = {}
        epoch_val_losses = {}
        
        for i, epoch in enumerate(self.epochs):
            if epoch not in epoch_train_losses:
                epoch_train_losses[epoch] = []
                epoch_val_losses[epoch] = []
            
            epoch_train_losses[epoch].append(self.train_losses[i])
            if i < len(self.val_losses):
                epoch_val_losses[epoch].append(self.val_losses[i])
        
        epochs_list = sorted(epoch_train_losses.keys())
        avg_train_losses = [np.mean(epoch_train_losses[e]) for e in epochs_list]
        avg_val_losses = [np.mean(epoch_val_losses[e]) if epoch_val_losses[e] else 0 for e in epochs_list]
        
        # 清除之前的绘图
        self.ax3.clear()
        self.ax3.set_title('Loss per Epoch', fontweight='bold')
        self.ax3.set_xlabel('Epochs')
        self.ax3.set_ylabel('Avg Loss')
        self.ax3.grid(True, alpha=0.3)
        
        # 绘制柱状图
        x = np.arange(len(epochs_list))
        width = 0.35
        
        bars1 = self.ax3.bar(x - width/2, avg_train_losses, width, label='Train Loss', alpha=0.8, color='skyblue')
        if any(avg_val_losses):
            bars2 = self.ax3.bar(x + width/2, avg_val_losses, width, label='Val Loss', alpha=0.8, color='lightcoral')
        
        self.ax3.set_xticks(x)
        self.ax3.set_xticklabels([f'Epoch {e}' for e in epochs_list])
        self.ax3.legend()
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    def update_stats_display(self):
        """更新统计信息显示"""
        self.ax4.clear()
        self.ax4.axis('off')
        
        # 计算统计信息
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        
        if len(self.train_losses) > 0:
            current_train_loss = self.train_losses[-1]
            avg_train_loss = np.mean(list(self.train_losses))
        else:
            current_train_loss = 0
            avg_train_loss = 0
        
        if len(self.val_losses) > 0:
            current_val_loss = self.val_losses[-1]
            avg_val_loss = np.mean(list(self.val_losses))
        else:
            current_val_loss = 0
            avg_val_loss = 0
        
        current_step = self.steps[-1] if self.steps else 0
        current_epoch = self.epochs[-1] if self.epochs else 0
        
        # 显示统计信息
        stats_text = f"""
Training Statistics:

Time: {str(elapsed_time).split('.')[0]}
Steps: {current_step:,}
Epoch: {current_epoch}

Train Loss: {current_train_loss:.6f}
Avg Train Loss: {avg_train_loss:.6f}

Val Loss: {current_val_loss:.6f}
Avg Val Loss: {avg_val_loss:.6f}

Best Val Loss: {self.best_val_loss:.6f}
Best Epoch: {self.best_epoch}

Log Dir: {self.output_dir}
        """
        
        self.ax4.text(0.05, 0.95, stats_text, transform=self.ax4.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    def save_training_plot(self, filename="training_progress.png"):
        """保存训练图表"""
        if self.enable_realtime:
            save_path = self.output_dir / filename
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 训练图表已保存: {save_path}")
    
    def save_training_data(self, filename="training_data.json"):
        """保存训练数据"""
        data = {
            "steps": list(self.steps),
            "epochs": list(self.epochs),
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
            "learning_rates": list(self.learning_rates),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "training_time": str(datetime.now() - self.start_time),
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat()
        }
        
        save_path = self.output_dir / filename
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ 训练数据已保存: {save_path}")
    
    def create_summary_report(self):
        """创建训练总结报告"""
        # 创建详细的统计图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Summary Report', fontsize=16, fontweight='bold')
        
        # 1. 损失曲线
        axes[0, 0].plot(list(self.steps), list(self.train_losses), 'b-', label='Train Loss', linewidth=2)
        if self.val_losses:
            axes[0, 0].plot(list(self.steps)[-len(self.val_losses):], list(self.val_losses), 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 学习率曲线
        if self.learning_rates:
            axes[0, 1].plot(list(self.steps)[-len(self.learning_rates):], list(self.learning_rates), 'g-', linewidth=2)
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Rate')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 损失分布直方图
        axes[0, 2].hist(list(self.train_losses), bins=30, alpha=0.7, label='Train Loss', color='blue')
        if self.val_losses:
            axes[0, 2].hist(list(self.val_losses), bins=30, alpha=0.7, label='Val Loss', color='red')
        axes[0, 2].set_title('Loss Distribution')
        axes[0, 2].set_xlabel('Loss')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        
        # 4. 训练进度
        if len(self.train_losses) > 10:
            # 计算移动平均
            window_size = min(50, len(self.train_losses) // 10)
            train_losses_array = np.array(list(self.train_losses))
            moving_avg = np.convolve(train_losses_array, np.ones(window_size)/window_size, mode='valid')
            
            axes[1, 0].plot(list(self.steps), list(self.train_losses), 'b-', alpha=0.3, label='Raw Loss')
            axes[1, 0].plot(list(self.steps)[window_size-1:], moving_avg, 'b-', linewidth=2, label=f'Moving Avg({window_size})')
            axes[1, 0].set_title('Smoothed Loss')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 性能指标
        axes[1, 1].axis('off')
        training_time = datetime.now() - self.start_time
        
        # 安全处理可能为空的数据
        final_train_loss = self.train_losses[-1] if self.train_losses else 0
        final_val_loss = self.val_losses[-1] if self.val_losses else None
        avg_train_loss = np.mean(list(self.train_losses)) if self.train_losses else 0
        avg_val_loss = np.mean(list(self.val_losses)) if self.val_losses else None
        
        if self.train_losses and len(self.train_losses) > 1:
            loss_improvement = self.train_losses[0] - self.train_losses[-1]
            improvement_pct = (loss_improvement / self.train_losses[0] * 100) if self.train_losses[0] != 0 else 0
        else:
            loss_improvement = 0
            improvement_pct = 0
        
        # 格式化可选值
        final_val_loss_str = f"{final_val_loss:.6f}" if final_val_loss is not None else "N/A"
        avg_val_loss_str = f"{avg_val_loss:.6f}" if avg_val_loss is not None else "N/A"
        
        metrics_text = f"""
Performance Metrics:

Time: {str(training_time).split('.')[0]}
Total Steps: {len(self.steps):,}
Epochs: {max(self.epochs) if self.epochs else 0}

Final Train Loss: {final_train_loss:.6f}
Final Val Loss: {final_val_loss_str}

Best Val Loss: {self.best_val_loss:.6f}
Best Epoch: {self.best_epoch}

Avg Train Loss: {avg_train_loss:.6f}
Avg Val Loss: {avg_val_loss_str}

Improvement: {loss_improvement:.6f}
Improvement %: {improvement_pct:.2f}%
        """
        
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 6. 训练建议
        axes[1, 2].axis('off')
        
        # 基于训练结果给出建议
        if len(self.train_losses) > 10:
            recent_train_loss = np.mean(list(self.train_losses)[-10:])
            early_train_loss = np.mean(list(self.train_losses)[:10])
            
            if recent_train_loss > early_train_loss * 0.9:
                suggestion = "🤔 Suggestion:\n• Loss decreasing slowly\n• Consider adjusting LR\n• Train longer"
            elif self.val_losses and len(self.val_losses) > 5:
                if self.val_losses[-1] > min(self.val_losses) * 1.1:
                    suggestion = "⚠️  Suggestion:\n• Possible overfitting\n• Consider early stopping\n• Increase regularization"
                else:
                    suggestion = "✅ Suggestion:\n• Good progress\n• Continue training"
            else:
                suggestion = "📈 Suggestion:\n• Training proceeding well\n• Monitor validation loss"
        else:
            suggestion = "ℹ️  Suggestion:\n• Just started\n• Keep monitoring"
        
        axes[1, 2].text(0.05, 0.95, suggestion, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存报告
        report_path = self.output_dir / "training_summary_report.png"
        fig.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"✅ 训练总结报告已保存: {report_path}")
        
        # 如果是实时模式，在新窗口显示
        if self.enable_realtime:
            plt.figure()
            plt.show()
        
        return fig
    
    def close(self):
        """关闭可视化器"""
        if self.enable_tensorboard:
            self.tb_writer.close()
        
        if self.enable_realtime:
            plt.ioff()
            plt.close('all')
        
        # 保存最终数据和图表
        self.save_training_data()
        self.save_training_plot()
        self.create_summary_report()
        
        print(f"✅ 训练可视化已保存到: {self.output_dir}")

# 使用示例
def demo_visualizer():
    """演示可视化器使用"""
    vis = TrainingVisualizer(enable_realtime=True, enable_tensorboard=True)
    
    # 模拟训练过程
    for epoch in range(3):
        for step in range(100):
            # 模拟训练数据
            train_loss = 2.0 * np.exp(-step/50) + 0.1 * np.random.random()
            val_loss = train_loss + 0.1 * np.random.random()
            lr = 5e-5 * (0.95 ** (step // 10))
            
            vis.log_step(step + epoch * 100, epoch + 1, train_loss, val_loss, lr)
            
            if step % 10 == 0:
                vis.update_plot()
                time.sleep(0.1)  # 模拟训练时间
    
    vis.close()

if __name__ == "__main__":
    demo_visualizer()