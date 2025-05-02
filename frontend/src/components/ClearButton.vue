<template>
  <div class="flex flex-row gap-5">
    <n-button type="error" size="large" @click="showModalRef = true">清空对话</n-button>

    <n-upload ref="uploadRef" @change="handleUpload" :multiple="false" :max="1" accept=".json">
      <n-button type="primary" size="large">载入对话</n-button>
    </n-upload>

  </div>

  <n-modal v-model:show="showModalRef" :mask-closable="false" preset="dialog" title="确认要清空所有对话？" content=""
    positive-text="确认" negative-text="取消" @positive-click="onPositiveClick" @negative-click="onNegativeClick" />
</template>

<script setup>
import { NButton, NModal, NUpload, useMessage } from "naive-ui";
import { useDialogStore } from "@/stores/dialog";
import { useThreadStore } from '../stores/result';
import { useNERStore } from "../stores/result";

import { ref } from "vue";
const uploadRef = ref(null)

const showModalRef = ref(false)

const dialogStore = useDialogStore();
const threadStore = useThreadStore();
const nerStore = useNERStore();

const clearAll = () => {
  dialogStore.clear();
  threadStore.clear();
  nerStore.clear();
}

const onNegativeClick = () => {
  showModalRef.value = false
}

const onPositiveClick = () => {
  clearAll()
  message.success('清空成功')
  showModalRef.value = false
}

const message = useMessage()
// upload dialog file.
const handleUpload = async (options) => {
  // read json file and update dialog store.
  const file = options.file.file;
  if (!file) {
    return;
  }
  const reader = new FileReader();
  reader.readAsText(file, 'utf-8');
  reader.onload = (e) => {
    const content = e.target?.result;
    if (typeof content !== 'string') {
      message.error('文件上传失败');
      return;
    }
    const obj = JSON.parse(content);
    if (obj.length == 0) {
      message.error('文件内容为空');
      return;
    }
    //rearrange id for each dialog
    let format_ok = true;
    obj.every((dialog, index) => {
      if (!dialog.name || !dialog.messages) {
        message.error('json文件格式错误，缺少name或messages字段');
        format_ok = false;
        return;
      }
      dialog.id = index;
      return true;
    });
    if (format_ok)
      dialogStore.setNewDialogs(obj);
    uploadRef.value?.clear();
  };
}

</script>