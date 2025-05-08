<template>
  <div class="flex flex-col gap-5">
    <NInput placeholder="输入微信联系人，回车提取对话到后端" @change="uploadDialog" class="w-1/2" />
    <NDropdown trigger="hover" :options="personOptions" @select="getDialog">
      <NButton>载入对话</NButton>
    </NDropdown>
    <n-button type="error" size="large" @click="showModalRef = true">清空对话</n-button>
  </div>
  <!-- add a global spin, display loading content-->
  <div v-show="isLoadingDialog" class="absolute inset-0 bg-black/40 z-50 flex justify-center items-center">
    <div class=" flex flex-col gap-6 justify-center">
      <NSpin size="large" />
      <div>{{ currentLoadingDialog }}</div>
    </div>
  </div>
  <n-modal v-model:show="showModalRef" :mask-closable="false" preset="dialog" title="确认要清空所有对话？" content=""
    positive-text="确认" negative-text="取消" @positive-click="onPositiveClick" @negative-click="onNegativeClick" />
</template>

<script setup>
import { NButton, NDropdown, NInput, NModal, NSpin, useMessage } from "naive-ui";
import { useDialogStore } from "@/stores/dialog";
import { useThreadStore } from '../stores/result';
import { useNERStore } from "../stores/result";
import { ref, onMounted } from "vue";

const showModalRef = ref(false)
const isLoadingDialog = ref(false)
const currentLoadingDialog = ref('正在提取对话...')
const personOptions = ref([])

const dialogStore = useDialogStore();
const threadStore = useThreadStore();
const nerStore = useNERStore();

const renewPersonOptions = async () => {
  // get all contact person options from backend
  return fetch(import.meta.env.VITE_BACKEND_URL + "/view_chats").then((res) => {
    if (res.status == 200)
      return res.json()
    else throw new Error(res.statusText)
  }).then((data) => {
    personOptions.value = data.chats.map((item) => {
      return {
        label: item,
        key: item
      }
    })
    message.success('获取联系人成功！')
  }).catch((err) => {
    message.error(err.toString())
  })
}

onMounted(renewPersonOptions)

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

const uploadDialog = async (name) => {
  if(name == null || name == '') {
    message.warning('请输入联系人')
    return
  }
  isLoadingDialog.value = true
  currentLoadingDialog.value = '正在从微信提取对话...'
  fetch(import.meta.env.VITE_BACKEND_URL + "/add_chat", {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ name })
  }).then((res) => {
    if (res.status == 200)
      return res.json()
    else throw new Error(res.statusText)
  }).then((data) => {
    message.success(data.result)
    renewPersonOptions()
  }).catch((err) => {
    console.log(err)
    message.error(err.toString())
  }).finally(() => {
    isLoadingDialog.value = false
  })
}

const getDialog = async (key) => {
  isLoadingDialog.value = true
  currentLoadingDialog.value = '正在做NER实体提取..'
  const backend_url = import.meta.env.VITE_BACKEND_URL
  const post_config = { method: 'POST', headers: { 'Content-Type': 'application/json' } }
  fetch(backend_url + "/perform_entity_recognition", post_config
  ).then(res => {
    if (res.status == 200) {
      currentLoadingDialog.value = '正在分析对话相关性..'
      return fetch(backend_url + "/analyze_relationships", post_config)
    } else throw new Error("NER 失败:" + res.statusText)
  }).then(res => {
    if (res.status == 200) {
      currentLoadingDialog.value = '正在获取对话..'
      return fetch(backend_url + "/get_chat_by_name", {...post_config,
        body: JSON.stringify({ name: key })
      })
    } else throw new Error("相关性分析失败：" + res.statusText)
  }).then(res => {
    if (res.status == 200) {
      return res.json()
    } else throw new Error(res.statusText)
  }).then(data => {
    if (data) {
      clearAll()
      console.log(data)
      const new_dialogs = []
      const new_ners = []
      const new_threads = []
      for (let sentence of data) {
        new_dialogs.push({
          id: sentence.id,
          name: sentence.sender,
          content: sentence.message,
          related: [...sentence.related_before, ...sentence.related_after]
        })
        if (sentence.entities && sentence.entities.length > 0) {
          for (let ner of sentence.entities) {
            new_ners.push({
              type: ner.entity.toLowerCase(),
              messageID: sentence.id,
              start: ner.range[0],
              end: ner.range[1],
              text: sentence.message.substring(ner.range[0], ner.range[1]),
            })
          }
        }
        // prevent undefined
        if (sentence.topic_start) {
          new_threads.push({
            id: new_threads.length,
            abstract: `话题：${new_threads.length}，关联度：${sentence.topic_Tightness.toFixed(2)}`,
            messageIDs: [sentence.id, ...(sentence.related_after.map(item => item.id))],
          })
        } else {
          new_threads.some((item) => {
            const find = sentence.related_before.map(item => item.id).includes(item.messageIDs[0])
            if (find) item.messageIDs.push(sentence.id)
            return find
          })
        }
      }
      if (new_ners.length > 0)
        nerStore.setNewNers(new_ners)
      if (new_dialogs.length > 0)
        dialogStore.setNewDialogs(new_dialogs)
      if (new_threads.length > 0)
        threadStore.setNewThreads(new_threads)
      message.success('获取对话成功！')
    }
  }).catch((err) => {
    console.log(err)
    message.error(err.toString())
  }).finally(() => {
    isLoadingDialog.value = false
  })
}

// upload dialog file.
// const handleUpload = async (options) => {
//   // read json file and update dialog store.
//   const file = options.file.file;
//   if (!file) {
//     return;
//   }
//   const reader = new FileReader();
//   reader.readAsText(file, 'utf-8');
//   reader.onload = (e) => {
//     const content = e.target?.result;
//     if (typeof content !== 'string') {
//       message.error('文件上传失败');
//       return;
//     }
//     const obj = JSON.parse(content);
//     if (obj.length == 0) {
//       message.error('文件内容为空');
//       return;
//     }
//     //rearrange id for each dialog
//     let format_ok = true;
//     obj.every((dialog, index) => {
//       if (!dialog.name || !dialog.messages) {
//         message.error('json文件格式错误，缺少name或messages字段');
//         format_ok = false;
//         return;
//       }
//       dialog.id = index;
//       return true;
//     });
//     if (format_ok)
//       dialogStore.setNewDialogs(obj);
//     uploadRef.value?.clear();
//   };
// }

</script>