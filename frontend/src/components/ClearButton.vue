<template>
  <div class="flex flex-col gap-5">
    <NInput placeholder="输入微信联系人，回车提取对话到后端" @change="uploadDialog" class="w-1/2" />
    <NList class=" w-full p-4">
      <NListItem v-for="(item, index) in personOptions" :key="`person_${index}`">
        <div class=" flex flex-row justify-between items-center gap-4">
          <div @click="getDialog(item.key)" class=" grow cursor-pointer p-2 hover:bg-gray-200 transition-colors">
            {{ item.label }}
          </div>
          <NButton strong secondary circle type="error" @click="deleteDialog(item.key)">
            <template #icon>
              <NIcon><svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
                  x="0px" y="0px" viewBox="0 0 512 512" enable-background="new 0 0 512 512" xml:space="preserve">
                  <g>
                    <path d="M128,405.429C128,428.846,147.198,448,170.667,448h170.667C364.802,448,384,428.846,384,405.429V160H128V405.429z M416,96
		h-80l-26.785-32H202.786L176,96H96v32h320V96z"></path>
                  </g>
                </svg></NIcon>
            </template>
          </NButton>
        </div>
      </NListItem>
    </NList>
    <n-button type="error" size="large" @click="showModalRef = true">清空对话</n-button>
  </div>
  <!-- add a global spin, display loading content-->
  <div v-show="isLoadingDialog" class="absolute inset-0 bg-black/50 z-50 flex justify-center items-center">
    <div class=" flex flex-col gap-6 justify-center">
      <NSpin size="large" />
      <div class=" text-white">{{ currentLoadingDialog }}</div>
    </div>
  </div>
  <n-modal v-model:show="showModalRef" :mask-closable="false" preset="dialog" title="确认要清空所有对话？" content=""
    positive-text="确认" negative-text="取消" @positive-click="onPositiveClick" @negative-click="onNegativeClick" />
</template>

<script setup>
import { NButton, NIcon, NInput, NList, NListItem, NModal, NSpin, useMessage } from "naive-ui";
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

const deleteDialog = async (key) => {
  // delete dialog from backend
  isLoadingDialog.value = true
  currentLoadingDialog.value = '正在删除联系人...'
  const backend_url = import.meta.env.VITE_BACKEND_URL
  const post_config = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ name: key })
  }
  fetch(backend_url + "/delete_chat", post_config).then((res) => {
    if (res.status == 200) {
      return renewPersonOptions()
    } else throw new Error(res.statusText)
  }).then(() => {
    message.success('删除联系人成功！')
  }).catch((err) => {
    console.error(err)
    message.error(err.toString())
  }).finally(() => {
    isLoadingDialog.value = false
  })
}

const renewPersonOptions = async () => {
  // get all contact person options from backend
  return fetch(import.meta.env.VITE_BACKEND_URL + "/view_chats").then((res) => {
    if (res.status == 200)
      return res.json()
    else throw new Error(res.statusText)
  }).then((data) => {
    if (data == null || data.chats == null) {
      personOptions.value = []
      throw new Error('获取联系人失败')
    }
    personOptions.value = data.chats.map((item) => {
      return {
        label: item,
        key: item
      }
    })
  }).catch((err) => {
    console.error(err)
    message.error(err.toString())
  })
}

onMounted(() => {
  isLoadingDialog.value = true
  currentLoadingDialog.value = '正在获取联系人列表...'
  renewPersonOptions().finally(() => {
    isLoadingDialog.value = false
  })
})

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
  if (name == null || name == '') {
    message.warning('请输入联系人')
    return
  }
  isLoadingDialog.value = true
  currentLoadingDialog.value = '正在上传联系人名称'
  const backend_url = import.meta.env.VITE_BACKEND_URL
  const post_config = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ name })
  }

  fetch(backend_url + "/add_chat", post_config).then((res) => {
    if (res.status == 200) {
      currentLoadingDialog.value = '正在从微信客户端提取对话，鼠标跳动是正常现象...'
      return fetch(backend_url + "/update_all_chats", post_config)
    }
    else throw new Error(res.statusText)
  }).then(res => {
    if (res.status == 200)
      return res.json()
    else throw new Error(res.statusText)
  }).then((data) => {
    message.success('载入联系人成功')
    renewPersonOptions()
  }).catch((err) => {
    console.error(err)
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
  fetch(backend_url + "/get_chat_by_name", {
    ...post_config,
    body: JSON.stringify({ name: key })
  }).then(res => {
    if (res.status == 200) {
      return res.json()
    } else throw new Error(res.statusText)
  }).then(data => {
    if (data && data.length > 0 && 
      data[0].entities && data[0].topic_start) {
      // if not have data[0].entities, then do ner
      return data
    } else return fetch(backend_url + "/perform_entity_recognition", post_config
    ).then(res => {
      if (res.status == 200) {
        currentLoadingDialog.value = '正在分析对话相关性..'
        return fetch(backend_url + "/analyze_relationships", post_config)
      } else throw new Error("NER 失败:" + res.statusText)
    }).then(res => {
      if (res.status == 200) {
        currentLoadingDialog.value = "正在抽取话题.."
        return fetch(backend_url + "/analyze_topics", post_config)
      } else throw new Error("相关性分析失败：" + res.statusText)
    }).then(res =>{
      if(res.status == 200){
      currentLoadingDialog.value = '正在获取对话..'
        return fetch(backend_url + "/get_chat_by_name", {
          ...post_config,
          body: JSON.stringify({ name: key })
        })
      } else throw new Error("话题提取失败：" + res.statusText)
    }).then(res => {
      if (res.status == 200) {
        return res.json()
      } else throw new Error(res.statusText)
    })
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
    console.error(err)
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